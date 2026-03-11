import torch
import torch.nn as nn

import backbones
from semantic_loader import load_semantic_embeddings
from semantic_modules import ProjectionHead
from semantic_losses import semantic_logits, source_semantic_alignment_loss, target_semantic_consistency_loss
from transfer_losses import TransferLoss


class NewTransferNet(nn.Module):
    def __init__(
        self,
        num_class,
        base_net='rsp_resnet50',
        transfer_loss='mmd',
        use_bottleneck=True,
        bottleneck_width=256,
        max_iter=1000,
        input_channels=3,
        use_semantic_branch=False,
        semantic_path='',
        semantic_dim=0,
        shared_dim=128,
        semantic_hidden_dim=0,
        semantic_conf_threshold=0.9,
        semantic_metric='cosine',
        semantic_logit_scale=16.0,
        semantic_normalize=True,
        semantic_src_weight=1.0,
        semantic_tgt_weight=1.0,
        semantic_tgt_warmup_epochs=0,
        semantic_margin_threshold=0.05,
        semantic_decay_power=1.0,
        semantic_tgt_ramp_power=1.0,
        semantic_beta=0.5,
        n_epoch=100,
        debug_semantic=False,
        **kwargs,
    ):
        super(NewTransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net, input_channels=input_channels)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss

        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        self.classifier_layer = nn.Linear(feature_dim, num_class)

        transfer_loss_args = {
            'loss_type': self.transfer_loss,
            'max_iter': max_iter,
            'num_class': num_class,
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Semantic branch
        self.use_semantic_branch = use_semantic_branch
        self.semantic_conf_threshold = semantic_conf_threshold
        self.semantic_metric = semantic_metric
        self.semantic_logit_scale = semantic_logit_scale
        self.semantic_src_weight = semantic_src_weight
        self.semantic_tgt_weight = semantic_tgt_weight
        self.semantic_tgt_warmup_epochs = semantic_tgt_warmup_epochs
        self.semantic_margin_threshold = semantic_margin_threshold
        self.semantic_decay_power = semantic_decay_power
        self.semantic_tgt_ramp_power = semantic_tgt_ramp_power
        self.semantic_beta = semantic_beta
        self.n_epoch = max(1, n_epoch)
        self.debug_semantic = debug_semantic

        self.visual_projector = None
        self.semantic_projector = None

        if self.use_semantic_branch:
            semantic_bank = load_semantic_embeddings(
                semantic_path=semantic_path,
                num_class=num_class,
                normalize=semantic_normalize,
            )
            # allow optional explicit semantic_dim but keep compatibility if mismatch
            if semantic_dim and semantic_dim > 0 and semantic_bank.shape[1] != semantic_dim:
                raise ValueError(
                    f'semantic_dim mismatch: expected {semantic_dim}, got {semantic_bank.shape[1]} from semantic bank'
                )
            semantic_dim = semantic_bank.shape[1]
            self.register_buffer('semantic_bank_coarse', semantic_bank)
            self.register_buffer('semantic_bank_fine', semantic_bank)

            self.visual_projector = ProjectionHead(feature_dim, shared_dim, hidden_dim=semantic_hidden_dim)
            self.semantic_projector = ProjectionHead(semantic_dim, shared_dim, hidden_dim=semantic_hidden_dim)

    def _source_label_and_bank(self, source_label, projected_bank):
        if source_label.min().item() >= 1 and projected_bank.shape[0] > 1:
            aligned_labels = source_label - 1
            bank = projected_bank[1:]
            if aligned_labels.max().item() >= bank.shape[0]:
                raise ValueError(
                    f"Source label/prototype mismatch after background shift: "
                    f"max(label-1)={aligned_labels.max().item()}, prototypes={bank.shape[0]}"
                )
            return aligned_labels, bank

        if source_label.max().item() >= projected_bank.shape[0]:
            raise ValueError(
                f"Source label/prototype mismatch: max(label)={source_label.max().item()}, "
                f"prototypes={projected_bank.shape[0]}"
            )
        return source_label, projected_bank

    def _semantic_decay(self, epoch):
        progress = min(max((epoch - 1) / max(1, self.n_epoch - 1), 0.0), 1.0)
        return (1.0 - progress) ** self.semantic_decay_power

    def _target_ramp(self, epoch):
        if epoch <= self.semantic_tgt_warmup_epochs:
            return 0.0
        remain = max(1, self.n_epoch - self.semantic_tgt_warmup_epochs)
        progress = min(max((epoch - self.semantic_tgt_warmup_epochs) / remain, 0.0), 1.0)
        return progress ** self.semantic_tgt_ramp_power

    def _semantic_forward(self, source_feat, target_feat, source_label, target_clf, epoch=1):
        zero = torch.tensor(0.0, device=source_feat.device)
        metrics = {
            'sem_loss': zero,
            'sem_src_loss': zero,
            'sem_tgt_loss': zero,
            'sem_valid_ratio': zero,
            'sem_conf_pass_ratio': zero,
            'sem_margin_pass_ratio': zero,
            'sem_src_coarse_loss': zero,
            'sem_src_fine_loss': zero,
            'lambda_sem_t': zero,
            'lambda_sem_tgt': zero,
            'source_coarse_sem_logits_shape': 'N/A',
            'source_fine_sem_logits_shape': 'N/A',
        }
        if not self.use_semantic_branch:
            return metrics

        main_source_feat = source_feat
        main_target_feat = target_feat

        zv_source = self.visual_projector(main_source_feat)
        zv_target = self.visual_projector(main_target_feat)
        zs_coarse = self.semantic_projector(self.semantic_bank_coarse)
        zs_fine = self.semantic_projector(self.semantic_bank_fine)

        src_labels_coarse, zs_coarse_aligned = self._source_label_and_bank(source_label, zs_coarse)
        src_labels_fine, zs_fine_aligned = self._source_label_and_bank(source_label, zs_fine)

        coarse_sem_logits = semantic_logits(
            zv_source,
            zs_coarse_aligned,
            metric=self.semantic_metric,
            logit_scale=self.semantic_logit_scale,
        )
        fine_sem_logits = semantic_logits(
            zv_source,
            zs_fine_aligned,
            metric=self.semantic_metric,
            logit_scale=self.semantic_logit_scale,
        )

        sem_src_coarse = source_semantic_alignment_loss(
            zv_source,
            src_labels_coarse,
            zs_coarse_aligned,
            metric=self.semantic_metric,
            logit_scale=self.semantic_logit_scale,
        )
        sem_src_fine = source_semantic_alignment_loss(
            zv_source,
            src_labels_fine,
            zs_fine_aligned,
            metric=self.semantic_metric,
            logit_scale=self.semantic_logit_scale,
        )
        sem_src_loss = (1.0 - self.semantic_beta) * sem_src_coarse + self.semantic_beta * sem_src_fine

        tgt_sem_logits = semantic_logits(
            zv_target,
            zs_fine,
            metric=self.semantic_metric,
            logit_scale=self.semantic_logit_scale,
        )

        sem_tgt_loss = zero
        sem_conf_pass_ratio = zero
        sem_margin_pass_ratio = zero
        sem_valid_ratio = zero
        if epoch > self.semantic_tgt_warmup_epochs:
            sem_tgt_loss, sem_conf_pass_ratio, sem_margin_pass_ratio, sem_valid_ratio = target_semantic_consistency_loss(
                target_clf,
                tgt_sem_logits,
                conf_threshold=self.semantic_conf_threshold,
                margin_threshold=self.semantic_margin_threshold,
            )

        lambda_sem_t = torch.tensor(self._semantic_decay(epoch), device=source_feat.device)
        lambda_sem_tgt = torch.tensor(
            min(self._target_ramp(epoch), self._semantic_decay(epoch)),
            device=source_feat.device,
        )

        sem_loss = (
            self.semantic_src_weight * lambda_sem_t * sem_src_loss
            + self.semantic_tgt_weight * lambda_sem_tgt * sem_tgt_loss
        )
        metrics['sem_loss'] = sem_loss
        metrics['sem_src_loss'] = sem_src_loss
        metrics['sem_tgt_loss'] = sem_tgt_loss
        metrics['sem_valid_ratio'] = sem_valid_ratio
        metrics['sem_conf_pass_ratio'] = sem_conf_pass_ratio
        metrics['sem_margin_pass_ratio'] = sem_margin_pass_ratio
        metrics['sem_src_coarse_loss'] = sem_src_coarse
        metrics['sem_src_fine_loss'] = sem_src_fine
        metrics['lambda_sem_t'] = lambda_sem_t
        metrics['lambda_sem_tgt'] = lambda_sem_tgt
        metrics['source_coarse_sem_logits_shape'] = tuple(coarse_sem_logits.shape)
        metrics['source_fine_sem_logits_shape'] = tuple(fine_sem_logits.shape)

        if self.debug_semantic:
            print(
                f"[semantic][epoch {epoch}] src coarse_logits={tuple(coarse_sem_logits.shape)} "
                f"fine_logits={tuple(fine_sem_logits.shape)} sem_src_coarse={sem_src_coarse.item():.4f} "
                f"sem_src_fine={sem_src_fine.item():.4f} sem_src={sem_src_loss.item():.4f} "
                f"conf_ratio={sem_conf_pass_ratio.item():.4f} margin_ratio={sem_margin_pass_ratio.item():.4f} "
                f"tgt_ratio={sem_valid_ratio.item():.4f} lambda_sem_t={lambda_sem_t.item():.4f} "
                f"lambda_sem_tgt={lambda_sem_tgt.item():.4f}"
            )

        return metrics

    def forward(self, source, target, source_label, epoch=1):
        source_outputs, \
        source_x_IN_1, source_x_1, source_x_style_1a, \
        source_x_IN_2, source_x_2, source_x_style_2a, \
        source_x_IN_3, source_x_3, source_x_style_3a = self.base_network(source)

        target_outputs, \
        target_x_IN_1, target_x_1, target_x_style_1a, \
        target_x_IN_2, target_x_2, target_x_style_2a, \
        target_x_IN_3, target_x_3, target_x_style_3a = self.base_network(target)

        if self.use_bottleneck:
            source_feat = self.bottleneck_layer(source_outputs)
            target_feat = self.bottleneck_layer(target_outputs)
        else:
            source_feat = source_outputs
            target_feat = target_outputs

        # classification loss
        source_clf = self.classifier_layer(source_feat)
        clf_loss = self.criterion(source_clf, source_label)

        # dis_loss
        dis_criterion = torch.nn.SmoothL1Loss()

        dis_content = dis_criterion(source_x_IN_1, source_x_1) \
                + dis_criterion(source_x_IN_2, source_x_2) \
                + dis_criterion(source_x_IN_3, source_x_3) \
                + dis_criterion(target_x_IN_1, target_x_1) \
                + dis_criterion(target_x_IN_2, target_x_2) \
                + dis_criterion(target_x_IN_3, target_x_3)
        dis_style = dis_criterion(source_x_style_1a, source_x_1) \
                    + dis_criterion(source_x_style_2a, source_x_2) \
                    + dis_criterion(source_x_style_3a, source_x_3) \
                    + dis_criterion(target_x_style_1a, target_x_1) \
                    + dis_criterion(target_x_style_2a, target_x_2) \
                    + dis_criterion(target_x_style_3a, target_x_3)
        dis_loss = dis_content - dis_style

        # transfer loss
        kwargs = {}
        if self.transfer_loss == 'lmmd':
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target_feat)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'daan':
            source_clf = self.classifier_layer(source_feat)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target_feat)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            target_clf = self.classifier_layer(target_feat)
            target_prob = nn.Softmax(dim=1)(target_clf)
        else:
            target_clf = self.classifier_layer(target_feat)

        transfer_target = target_prob if self.transfer_loss == 'bnm' else target_feat
        transfer_loss = self.adapt_loss(source_feat, transfer_target, **kwargs)
        semantic_metrics = self._semantic_forward(source_feat, target_feat, source_label, target_clf, epoch=epoch)

        return clf_loss, dis_loss, transfer_loss, semantic_metrics

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append({'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr})

        if self.use_semantic_branch:
            params.append({'params': self.visual_projector.parameters(), 'lr': 1.0 * initial_lr})
            params.append({'params': self.semantic_projector.parameters(), 'lr': 1.0 * initial_lr})

        # Loss-dependent
        if self.transfer_loss == 'adv':
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == 'daan':
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        x_4, \
        x_IN_1, x_1, x_style_1a, \
        x_IN_2, x_2, x_style_2a, \
        x_IN_3, x_3, x_style_3a = self.base_network(x)
        feat = self.bottleneck_layer(x_4) if self.use_bottleneck else x_4
        clf = self.classifier_layer(feat)
        return clf

    def get_features(self, x):
        x_4, \
        x_IN_1, x_1, x_style_1a, \
        x_IN_2, x_2, x_style_2a, \
        x_IN_3, x_3, x_style_3a = self.base_network(x)
        return self.bottleneck_layer(x_4) if self.use_bottleneck else x_4

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == 'daan':
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

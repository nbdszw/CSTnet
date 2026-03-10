import torch
import torch.nn as nn

import backbones
from semantic_loader import load_semantic_embeddings
from semantic_modules import ProjectionHead
from semantic_losses import source_semantic_alignment_loss, target_semantic_consistency_loss
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
        semantic_normalize=True,
        semantic_src_weight=1.0,
        semantic_tgt_weight=1.0,
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
        self.semantic_src_weight = semantic_src_weight
        self.semantic_tgt_weight = semantic_tgt_weight

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
            self.register_buffer('semantic_bank', semantic_bank)

            self.visual_projector = ProjectionHead(feature_dim, shared_dim, hidden_dim=semantic_hidden_dim)
            self.semantic_projector = ProjectionHead(semantic_dim, shared_dim, hidden_dim=semantic_hidden_dim)

    def _semantic_forward(self, source_feat, target_feat, source_label, target_clf):
        zero = torch.tensor(0.0, device=source_feat.device)
        metrics = {
            'sem_loss': zero,
            'sem_src_loss': zero,
            'sem_tgt_loss': zero,
            'sem_valid_ratio': zero,
        }
        if not self.use_semantic_branch:
            return metrics

        zv_source = self.visual_projector(source_feat)
        zv_target = self.visual_projector(target_feat)
        zs = self.semantic_projector(self.semantic_bank)

        sem_src_loss = source_semantic_alignment_loss(
            zv_source,
            source_label,
            zs,
            metric=self.semantic_metric,
        )
        sem_tgt_loss, sem_valid_ratio = target_semantic_consistency_loss(
            zv_target,
            target_clf,
            zs,
            conf_threshold=self.semantic_conf_threshold,
            metric=self.semantic_metric,
        )

        sem_loss = self.semantic_src_weight * sem_src_loss + self.semantic_tgt_weight * sem_tgt_loss
        metrics['sem_loss'] = sem_loss
        metrics['sem_src_loss'] = sem_src_loss
        metrics['sem_tgt_loss'] = sem_tgt_loss
        metrics['sem_valid_ratio'] = sem_valid_ratio
        return metrics

    def forward(self, source, target, source_label):
        source_outputs, \
        source_x_IN_1, source_x_1, source_x_style_1a, \
        source_x_IN_2, source_x_2, source_x_style_2a, \
        source_x_IN_3, source_x_3, source_x_style_3a = self.base_network(source)

        target_outputs, \
        target_x_IN_1, target_x_1, target_x_style_1a, \
        target_x_IN_2, target_x_2, target_x_style_2a, \
        target_x_IN_3, target_x_3, target_x_style_3a = self.base_network(target)

        if self.use_bottleneck:
            source = self.bottleneck_layer(source_outputs)
            target = self.bottleneck_layer(target_outputs)

        # classification loss
        source_clf = self.classifier_layer(source)
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
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'daan':
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            target_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(target_clf)
        else:
            target_clf = self.classifier_layer(target)

        transfer_loss = self.adapt_loss(source, target, **kwargs)
        semantic_metrics = self._semantic_forward(source, target, source_label, target_clf)

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
        x = self.bottleneck_layer(x_4)
        clf = self.classifier_layer(x)
        return clf

    def get_features(self, x):
        x_4, \
        x_IN_1, x_1, x_style_1a, \
        x_IN_2, x_2, x_style_2a, \
        x_IN_3, x_3, x_style_3a = self.base_network(x)
        return self.bottleneck_layer(x_4)

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == 'daan':
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

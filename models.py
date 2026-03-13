import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones

class NewTransferNet(nn.Module):
    def __init__(self, num_class, base_net='rsp_resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000, input_channels=3,**kwargs):
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
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

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
        """
        dis_content = dis_criterion(target_x_IN_1, target_x_1) \
                + dis_criterion(target_x_IN_2, target_x_2) \
                + dis_criterion(target_x_IN_3, target_x_3)
        dis_style = dis_criterion(target_x_style_1a, target_x_1) \
                    + dis_criterion(target_x_style_2a, target_x_2) \
                    + dis_criterion(target_x_style_3a, target_x_3)
        """
        dis_loss = dis_content - dis_style
        # dis_loss = dis_content
        # dis_loss = - dis_style

        """      
        # class-level alignment loss
        target_clf = self.classifier_layer(target)
        source_clf = self.classifier_layer(source)
        source_probs = torch.softmax(source_clf, dim=1)
        class_loss = torch.sum(source_probs * torch.nn.functional.cross_entropy(target_clf, source_label))
        class_loss = class_loss / source_probs.size(0) 
        """

        # transfer loss
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)
        
        transfer_loss = self.adapt_loss(source, target, **kwargs)

        return clf_loss, dis_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
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
    
    def get_features(self,x):
        x_4, \
        x_IN_1, x_1, x_style_1a, \
        x_IN_2, x_2, x_style_2a, \
        x_IN_3, x_3, x_style_3a = self.base_network(x)
        return self.bottleneck_layer(x_4)

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass
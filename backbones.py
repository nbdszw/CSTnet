import torch.nn as nn
from torchvision import models
import torch
from attention import simam_module

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "rsp_resnet50": models.resnet50,
    "rsp_resnet18": models.resnet18,
}

def get_backbone(name,input_channels=3):
    if "rsp_resnet" in name.lower():
        return RSPResNetBackbone(name,input_channels=input_channels)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()


class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type,input_channels = 3):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim
    
class RSPResNetBackbone(nn.Module):
    def __init__(self, network_type,input_channels = 3):
        super(RSPResNetBackbone, self).__init__()
        model_resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        for name, module in model_resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)

        self.avgpool = model_resnet.avgpool
        self._feature_dim = model_resnet.fc.in_features
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.SimAM = simam_module()

        if '18' in network_type:
            self.IN1 = nn.InstanceNorm2d(64, affine=True)
            self.IN2 = nn.InstanceNorm2d(128, affine=True)
            self.IN3 = nn.InstanceNorm2d(256, affine=True)

        elif '50' in network_type:
            self.IN1 = nn.InstanceNorm2d(256, affine=True)
            self.IN2 = nn.InstanceNorm2d(512, affine=True)
            self.IN3 = nn.InstanceNorm2d(1024, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_IN_1 = self.IN1(x_1)
        x_style_1 = x_1 - x_IN_1
        x_style_1a, _, _ = self.SimAM(x_style_1)
        x_1 = x_IN_1 + x_style_1a
        # x_1 = x_IN_1

        x_2 = self.layer2(x_1)
        x_IN_2 = self.IN2(x_2)
        x_style_2 = x_2 - x_IN_2
        x_style_2a, _, _ = self.SimAM(x_style_2)
        x_2 = x_IN_2 + x_style_2a
        # x_2 = x_IN_2

        x_3 = self.layer3(x_2)
        x_IN_3 = self.IN3(x_3)
        x_style_3 = x_3 - x_IN_3
        x_style_3a, _, _ = self.SimAM(x_style_3)
        x_3 = x_IN_3 + x_style_3a
        # x_3 = x_IN_3

        x_4 = self.layer4(x_3)
        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)

        return x_4, \
                x_IN_1, x_1, x_style_1a, \
                x_IN_2, x_2, x_style_2a, \
                x_IN_3, x_3, x_style_3a
    

    
    def output_num(self):
        return self._feature_dim
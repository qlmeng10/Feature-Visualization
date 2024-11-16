# Created by MengQingluo on 2024/11/12 18:29
import pdb
from collections import OrderedDict

from torch import nn
import torchvision

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        self.in_planes = 2048
        self.num_classes = 1000
        self.feature1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier256 = nn.Linear(256, self.num_classes, bias=False)
        self.classifier512 = nn.Linear(512, self.num_classes, bias=False)
        self.classifier1024 = nn.Linear(1024, self.num_classes, bias=False)

    def forward(self, x):
        # pdb.set_trace()
        feat = self.feature1(x)         # 64
        layer1 = self.layer1(feat)      # 256
        layer2 = self.layer2(layer1)    # 512
        layer3 = self.layer3(layer2)    # 1024
        layer4 = self.layer4(layer3)    # 2048

        return OrderedDict([["feat_res1", layer1], ["feat_res2", layer2], ["feat_res3", layer3], ["feat_res4", layer4]])


def build_resnet(cfg):
    resnet_model = torchvision.models.resnet.__dict__[cfg.MODEL.NAME](pretrained=True)
    # freeze layers
    resnet_model.conv1.weight.requires_grad_(False)
    resnet_model.bn1.weight.requires_grad_(False)
    resnet_model.bn1.bias.requires_grad_(False)
    return Backbone(resnet_model)


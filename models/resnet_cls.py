#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

def get_model(num_class, config_dict):
    ResNet(num_class, config_dict)

class ResNet(nn.Module):
    def __init__(self, num_class, config_dict):
        super(ResNet, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_class)

    def apply_transformations(self, input_tensor):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(input_tensor)

    def forward(self, x):
        # TODO transformations inlcuding fitting image size
        return self.model(x), None

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        loss = torch.nn.functional.nll_loss(pred, target)

        return loss
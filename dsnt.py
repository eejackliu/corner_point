import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision as tv
import random
import math
from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader as dataloader
# import nonechucks as nc
from data import my_data,label_acc_score,voc_colormap,seg_target
from dsntnn import dsnt
import dsntnn
image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
mask_transform=transforms.Compose([transforms.ToTensor()])
trainset=my_data(transform=image_transform,target_transform=mask_transform)
testset=my_data(image_set='test',transform=image_transform,target_transform=mask_transform)
trainload=torch.utils.data.DataLoader(trainset,batch_size=32)
testload=torch.utils.data.DataLoader(testset,batch_size=32)
# device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
resnet=tv.models.resnet34()

class Cordinate_Point(nn.Module):
    """Abstract base class for human pose estimation models."""

    def _hm_preact(self, x, preact):
        n_chans = x.size(-3)
        height = x.size(-2)
        width = x.size(-1)
        x = x.view(-1, height * width)
        if preact == 'softmax':
            x = nn.functional.softmax(x, dim=-1)
        # elif preact == 'thresholded_softmax':
        #     x = thresholded_softmax(x, -0.5)
        elif preact == 'abs':
            x = x.abs()
            x = x / (x.sum(-1, keepdim=True) + 1e-12)
        elif preact == 'relu':
            x = nn.functional.relu(x, inplace=False)
            x = x / (x.sum(-1, keepdim=True) + 1e-12)
        elif preact == 'sigmoid':
            x = nn.functional.sigmoid(x)
            x = x / (x.sum(-1, keepdim=True) + 1e-12)
        else:
            raise Exception('unrecognised heatmap preactivation function: {}'.format(preact))
        x = x.view(-1, n_chans, height, width)
        return x

    def _calculate_reg_loss(self, target_var, reg, hm_var, hm_sigma):
        # Convert sigma (aka standard deviation) from pixels to normalized units
        sigma = (2.0 * hm_sigma / hm_var.size(-1))

        # Apply a regularisation term relating to the shape of the heatmap.
        if reg == 'var':
            reg_loss = dsntnn.variance_reg_losses(hm_var, sigma, )
        elif reg == 'kl':
            reg_loss = dsntnn.kl_reg_losses(hm_var, target_var, sigma)
        elif reg == 'js':
            reg_loss = dsntnn.js_reg_losses(hm_var, target_var, sigma)
        # elif reg == 'mse':

            # reg_loss = dsntnn.mse_reg_losses(hm_var, target_var, sigma, mask_var)
        else:
            reg_loss = 0

        return reg_loss

class Hed(Cordinate_Point):
    def __init__(self,output_strat,dilate=2,preact='softmax',hm_sigma=1.0,reg_coeff=1.0):
        super(Hed,self).__init__()
        self.output_strat =output_strat
        self.hm_sigma=hm_sigma
        self.preact=preact
        self.reg_coeff = reg_coeff
        fcn_modules = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        ]
        layers = [ resnet.layer3, resnet.layer4]
        for i, layer in enumerate(layers[len(layers) - dilate:]):
            dilx = dily = 2 ** (i + 1)
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    if module.stride == (2, 2):
                        module.stride = (1, 1)
                    elif module.kernel_size == (3, 3):
                        kx, ky = module.kernel_size
                        module.dilation = (dilx, dily)
                        module.padding = ((dilx * (kx - 1) + 1) // 2, (dily * (ky - 1) + 1) // 2)
        truncate=0
        fcn_modules.extend(layers[:len(layers) - truncate])
        self.fcn = nn.Sequential(*fcn_modules)
        if truncate > 0:
            feats = layers[-truncate][0].conv1.in_channels
        else:
            feats = resnet.fc.in_features
        self.hm_conv = nn.Conv2d(feats, self.n_chans, kernel_size=1, bias=False)
    def forward_loss(self, out_var, target_var, mask_var):
        if self.output_strat == 'dsnt' or self.output_strat == 'fc':
            loss = dsntnn.euclidean_losses(out_var, target_var, )

            reg_loss = self._calculate_reg_loss(
                target_var, self.reg, self.heatmaps, self.hm_sigma)

            return loss + self.reg_coeff * reg_loss
        elif self.output_strat == 'fc':
            return dsntnn.euclidean_losses(out_var, target_var, )
        # elif self.output_strat == 'gauss':
        #     norm_coords = target_var.data.cpu()
        #     width = out_var.size(-1)
        #     height = out_var.size(-2)
        #
        #     target_hm = util.encode_heatmaps(norm_coords, width, height, self.hm_sigma)
        #     target_hm_var = Variable(target_hm.cuda())
        #
        #     loss = nn.functional.mse_loss(out_var, target_hm_var)
        #     return loss

        raise Exception('invalid configuration')
    def forward_part1(self, x):
        """Forward from images to unnormalized heatmaps"""

        x = self.fcn(x)
        x = self.hm_conv(x)
        return x

    def forward_part2(self, x):
        """Forward from unnormalized heatmaps to output"""

        if self.output_strat == 'dsnt':
            x = self._hm_preact(x, self.preact)
            self.heatmaps = x
            x = dsnt(x)
        # elif self.output_strat == 'fc':
        #     x = self._hm_preact(x, self.preact)
        #     self.heatmaps = x
        #     height = x.size(-2)
        #     width = x.size(-1)
        #     x = x.view(-1, height * width)
        #     x = self.out_fc(x)
        #     x = x.view(-1, self.n_chans, 2)
        # else:
        #     self.heatmaps = x

        return x
    def forward(self, *inputs):
        x = inputs[0]
        x = self.forward_part1(x)
        x = self.forward_part2(x)

        return x
model=Hed('dsnt')
epoch=30
for i in range( epoch):
    for data,label
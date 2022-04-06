import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

from ssim import MS_SSIM

import numpy as np

from CommonFunc import *

# Perception loss with pre-trained vgg net
class PerceptionLoss(nn.Module):
    def __init__(self, feature_layer=1, perception_perBand=False):
        # multi-layer perception loss can be calculated
        # mostly only the last feature layer is used (1)
        # if perception_perBand is true, the perception will be calculated band by band
        # if perception_perBand is false, the first three bands are used as RGB
        # for multi-spectral images, set perception_perBand to be True
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.net = vgg
        # multi-layer perception loss can be calculated
        feature_layer_list = [29, 22, 15, 8, 3]
        # feature_layer_list = [30, 23, 16, 9, 4]
        feature_layer = feature_layer if feature_layer > 0 else 1
        feature_layer = feature_layer if feature_layer < 6 else 5
        self.feature_layer_list = feature_layer_list[:feature_layer]
        self.perception_perBand = perception_perBand
        self.loss = nn.MSELoss()

    def forward(self, target_image, generate_image, cmask):
        perception_loss = 0
        if self.perception_perBand == False:
            assert target_image.shape[1] >= 3
            x = target_image[:, 0:3, :, :] * (1 - cmask.repeat((1, 3, 1, 1)))
            y = generate_image[:, 0:3, :, :] * (1 - cmask.repeat((1, 3, 1, 1)))
            layer_num = len(self.feature_layer_list)
            for i in range(len(self.net)):
                x = self.net[i](x)
                y = self.net[i](y)
                if i in self.feature_layer_list:
                    perception_loss += self.loss(x, y) / layer_num
        else:
            n_channels = target_image.shape[1]
            for b in range(n_channels):
                x = (target_image[:, b, :, :].unsqueeze(1) * (1 - cmask)).repeat((1, 3, 1, 1))
                y = (generate_image[:, b, :, :].unsqueeze(1) * (1 - cmask)).repeat((1, 3, 1, 1))
                layer_num = len(self.feature_layer_list)
                for i in range(len(self.net)):
                    x = self.net[i](x)
                    y = self.net[i](y)
                    if i in self.feature_layer_list:
                        perception_loss += self.loss(x, y) / layer_num / n_channels
        return perception_loss

# loss function for generator
class CNetLoss(nn.Module):
    def __init__(self, channel=4, perception_layer=1, perception_perBand=True):
        super(CNetLoss, self).__init__()
        self.mse = nn.MSELoss()
        # self.loss_generator = nn.MSELoss()
        self.loss_generator = nn.L1Loss()
        self.loss_perception = PerceptionLoss(feature_layer=perception_layer, perception_perBand=perception_perBand)
        self.ssim = MS_SSIM(data_range=1.0, channel=channel)

    def forward(self, target_image, generate_image, cmap, generator_mask_switch=False):
        # in cmap, higher value indicates higher probability to be changed
        cmask = (torch.sign(cmap - 0.5) + 1) / 2
        num_pixel = target_image.size()[2] * target_image.size()[3]
        num_wnc = torch.sum(1 - cmap, (1, 2, 3))
        target_image_mask = target_image * (1 - cmap.repeat((1, target_image.size()[1], 1, 1)))
        generate_image_mask = generate_image * (1 - cmap.repeat((1, generate_image.size()[1], 1, 1)))

        generator_loss = 0
        for i in range(target_image.shape[0]):
            generator_loss += self.loss_generator(target_image_mask[i], generate_image_mask[i]) * num_pixel / num_wnc[i]
        generator_loss = generator_loss / target_image.shape[0]

        # if generator_mask_switch is True, the cmp will be translated to binary change mask
        l1_loss = torch.mean(abs(cmap))
        if generator_mask_switch == True:
            perception_loss = self.loss_perception(target_image, generate_image, cmask)
        else:
            perception_loss = self.loss_perception(target_image, generate_image, cmap)

        ssim_loss = 1 - self.ssim(target_image_mask, generate_image_mask)

        return generator_loss, l1_loss, perception_loss, ssim_loss

# loss function for generator
# used in weakly supervised change detection task
# the only difference is that, there may be some images with all changed pixels
class CGeneratorLoss(nn.Module):
    def __init__(self, channel=3, perception_layer=1, perception_perBand=False):
        super(CGeneratorLoss, self).__init__()
        # self.loss_generator = nn.L1Loss()
        self.loss_generator = nn.MSELoss()
        self.ssim = MS_SSIM(data_range=1.0, channel=channel)
        self.loss_perception = PerceptionLoss(feature_layer=perception_layer, perception_perBand=perception_perBand)

    def forward(self, target_image, generate_image, cmap):
        num_pixel = target_image.size()[2] * target_image.size()[3]
        num_wnc = torch.sum(1 - cmap, (1, 2, 3))
        target_image_mask = target_image * (1 - cmap.repeat((1, target_image.size()[1], 1, 1)))
        generate_image_mask = generate_image * (1 - cmap.repeat((1, generate_image.size()[1], 1, 1)))

        generator_loss = 0
        for i in range(target_image.shape[0]):
            if num_wnc[i] == 0:
                continue
            generator_loss += self.loss_generator(target_image_mask[i], generate_image_mask[i]) * num_pixel / num_wnc[i]
        generator_loss = generator_loss / target_image.shape[0]
        ssim_loss = 1 - self.ssim(target_image_mask, generate_image_mask)

        perception_loss = self.loss_perception(target_image, generate_image, cmap)

        return generator_loss, ssim_loss, perception_loss

# Loss for regional supervised change detection
def region_loss(cmap, region, criterion):
    num_pixel = cmap.size()[2] * cmap.size()[3]
    num_region = torch.sum(region, (1, 2, 3))
    cmap = cmap * region
    ref = torch.zeros_like(region)
    # criterion = nn.BCELoss()

    r_loss = 0
    for i in range(cmap.shape[0]):
        if num_region[i] == 0:
            continue
        r_loss += criterion(cmap[i], ref[i]) * num_pixel / num_region[i]
    r_loss = r_loss / cmap.shape[0]

    return r_loss


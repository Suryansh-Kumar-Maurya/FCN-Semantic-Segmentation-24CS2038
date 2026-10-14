# Deep Learning 2026 (DL-029) - Paper Reproduction
# Roll No: 24CS2038 | Section: IDD
# Paper: Fully Convolutional Networks for Semantic Segmentation (Long, Shelhamer, Darrell, CVPR 2015)
# File: models/fcn.py

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    2D bilinear interpolation filter for initializing ConvTranspose2d layers.
    Following Long et al. (CVPR 2015), Section 3.3.
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    for i in range(in_channels):
        for j in range(out_channels):
            if i == j:
                weight[i, j, :, :] = filt
    return torch.from_numpy(weight)


class VGGBackbone(nn.Module):
    """
    VGG-16 backbone split into 5 pooling stages, with fc6 and fc7 
    converted into 2D convolutions (convolutionalization).
    """
    def __init__(self, pretrained=True):
        super(VGGBackbone, self).__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        vgg16 = models.vgg16(weights=weights)
        feats = list(vgg16.features.children())

        # Stage 1: downsample to H/2, W/2 (64 channels)
        self.stage1 = nn.Sequential(*feats[0:5])
        # Stage 2: downsample to H/4, W/4 (128 channels)
        self.stage2 = nn.Sequential(*feats[5:10])
        # Stage 3: downsample to H/8, W/8 (256 channels) - Pool3
        self.stage3 = nn.Sequential(*feats[10:17])
        # Stage 4: downsample to H/16, W/16 (512 channels) - Pool4
        self.stage4 = nn.Sequential(*feats[17:24])
        # Stage 5: downsample to H/32, W/32 (512 channels) - Pool5
        self.stage5 = nn.Sequential(*feats[24:31])

        # Convolutionalize fc6: 512*7*7 (25088) -> 4096
        # Kernel size 7x7 with padding 3 keeps spatial shape H/32 x W/32
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5)

        # Convolutionalize fc7: 4096 -> 4096 (1x1 conv)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5)

        if pretrained:
            self._load_vgg_fc_weights(vgg16)

    def _load_vgg_fc_weights(self, vgg16):
        # fc6 weights are [4096, 25088], reshape to [4096, 512, 7, 7]
        self.conv6.weight.data.copy_(vgg16.classifier[0].weight.data.view(4096, 512, 7, 7))
        self.conv6.bias.data.copy_(vgg16.classifier[0].bias.data)

        # fc7 weights are [4096, 4096], reshape to [4096, 4096, 1, 1]
        self.conv7.weight.data.copy_(vgg16.classifier[3].weight.data.view(4096, 4096, 1, 1))
        self.conv7.bias.data.copy_(vgg16.classifier[3].bias.data)


class FCN32s(nn.Module):
    """
    FCN-32s: Single-stream architecture.
    Direct 32x upsampling from conv7 using transposed convolution.
    """
    def __init__(self, num_classes=11, pretrained=True, init_bilinear=True):
        super(FCN32s, self).__init__()
        self.num_classes = num_classes
        self.backbone = VGGBackbone(pretrained=pretrained)
        
        # 1x1 conv to project 4096 features to C class scores
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # 32x upsample: s=32, k=64, p=16 gives (Hin - 1)*32 - 32 + 64 = 32*Hin
        self.upscore32 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False
        )

        if init_bilinear:
            self.upscore32.weight.data.copy_(
                get_upsampling_weight(num_classes, num_classes, 64)
            )

    def forward(self, x):
        h = self.backbone.stage1(x)
        h = self.backbone.stage2(h)
        h = self.backbone.stage3(h)
        h = self.backbone.stage4(h)
        h = self.backbone.stage5(h)

        h = self.backbone.drop6(self.backbone.relu6(self.backbone.conv6(h)))
        h = self.backbone.drop7(self.backbone.relu7(self.backbone.conv7(h)))

        score = self.score_fr(h)
        out = self.upscore32(score)
        return out


class FCN16s(nn.Module):
    """
    FCN-16s: Two-stream fusion.
    Combines conv7 (stride 32, upsampled 2x) with pool4 (stride 16),
    then upsamples 16x to original resolution.
    """
    def __init__(self, num_classes=11, pretrained=True, init_bilinear=True):
        super(FCN16s, self).__init__()
        self.num_classes = num_classes
        self.backbone = VGGBackbone(pretrained=pretrained)

        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        # 2x upsampling for conv7 scores
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )

        # 16x upsampling for fused pool4 + conv7 scores
        self.upscore16 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=32, stride=16, padding=8, bias=False
        )

        if init_bilinear:
            self.upscore2.weight.data.copy_(
                get_upsampling_weight(num_classes, num_classes, 4)
            )
            self.upscore16.weight.data.copy_(
                get_upsampling_weight(num_classes, num_classes, 32)
            )

    def forward(self, x):
        x1 = self.backbone.stage1(x)
        x2 = self.backbone.stage2(x1)
        x3 = self.backbone.stage3(x2)
        x4 = self.backbone.stage4(x3)   # pool4: stride 16, 512 channels
        x5 = self.backbone.stage5(x4)   # pool5: stride 32, 512 channels

        h = self.backbone.drop6(self.backbone.relu6(self.backbone.conv6(x5)))
        h = self.backbone.drop7(self.backbone.relu7(self.backbone.conv7(h)))

        score_fr = self.score_fr(h)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(x4)
        fuse_pool4 = upscore2 + score_pool4  # skip connection via element-wise sum

        out = self.upscore16(fuse_pool4)
        return out


class FCN8s(nn.Module):
    """
    FCN-8s: Three-stream fusion.
    Combines conv7 + pool4 + pool3 features for finer boundary recovery.
    """
    def __init__(self, num_classes=11, pretrained=True, init_bilinear=True):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes
        self.backbone = VGGBackbone(pretrained=pretrained)

        # 1x1 convs to map each feature scale to num_classes
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

        # 2x upsamplers
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )

        # 8x final upsampler
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False
        )

        if init_bilinear:
            self.upscore2.weight.data.copy_(
                get_upsampling_weight(num_classes, num_classes, 4)
            )
            self.upscore_pool4.weight.data.copy_(
                get_upsampling_weight(num_classes, num_classes, 4)
            )
            self.upscore8.weight.data.copy_(
                get_upsampling_weight(num_classes, num_classes, 16)
            )

    def forward(self, x):
        x1 = self.backbone.stage1(x)
        x2 = self.backbone.stage2(x1)
        x3 = self.backbone.stage3(x2)   # pool3: stride 8, 256 channels
        x4 = self.backbone.stage4(x3)   # pool4: stride 16, 512 channels
        x5 = self.backbone.stage5(x4)   # pool5: stride 32, 512 channels

        h = self.backbone.drop6(self.backbone.relu6(self.backbone.conv6(x5)))
        h = self.backbone.drop7(self.backbone.relu7(self.backbone.conv7(h)))

        score_fr = self.score_fr(h)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(x4)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(x3)
        fuse_pool3 = upscore_pool4 + score_pool3

        out = self.upscore8(fuse_pool3)
        return out


def build_fcn_model(model_name, num_classes=11, pretrained=True, init_bilinear=True):
    name = model_name.lower().replace("-", "").replace("_", "")
    if name == "fcn32s":
        return FCN32s(num_classes=num_classes, pretrained=pretrained, init_bilinear=init_bilinear)
    elif name == "fcn16s":
        return FCN16s(num_classes=num_classes, pretrained=pretrained, init_bilinear=init_bilinear)
    elif name == "fcn8s":
        return FCN8s(num_classes=num_classes, pretrained=pretrained, init_bilinear=init_bilinear)
    else:
        raise ValueError(f"Model {model_name} not recognized. Use fcn32s, fcn16s, or fcn8s.")

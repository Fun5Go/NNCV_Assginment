import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SqueezeExcite


# --------------------- Convolutional Block ---------------------
class conv_block(nn.Module):
    """
    A standard convolutional block with two convolution layers,
    batch normalization, ReLU activations, and optional dropout.
    """
    def __init__(self, in_c, out_c, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcite(out_c, rd_ratio=1/16)

    def forward(self, x):
        x = self.conv(x)
        return self.se(x)

# --------------------- ASPP Modules ---------------------
class ASPPConv(nn.Sequential):
    """
    ASPP convolutional block with dilation.
    """
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """
    Global average pooling block for ASPP.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]  # Get spatial dimensions
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    ASPP module with multiple parallel dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ),
            *[ASPPConv(in_channels, out_channels, d) for d in dilations[1:]],
            ASPPPooling(in_channels, out_channels),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(self.blocks) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, x):
        res = [block(x) for block in self.blocks]
        return self.project(torch.cat(res, dim=1))


# --------------------- Attention Gate ---------------------
class AttentionGate(nn.Module):
    """
    Attention gate to refine skip connections using gating signal.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# --------------------- Decoder Block ---------------------
class DecoderBlock(nn.Module):
    """
    Decoder block with optional attention gate and upsampling.
    """
    def __init__(self, in_c, skip_c, out_c, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            self.att = AttentionGate(F_g=out_c, F_l=skip_c, F_int=out_c // 2)
        self.conv = conv_block(skip_c + out_c, out_c, dropout=0.3)

    def forward(self, x, skip):
        x = self.up(x)
        if self.use_attention:
            skip = self.att(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# --------------------- Final Output Head ---------------------
class FinalHead(nn.Module):
    """
    Final output layer with global feature integration.
    """
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, n_classes, 1)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(x + global_feat)


# --------------------- Complete Model ---------------------
class Model(nn.Module):
    """
    Full U-Net-like encoder-decoder model with ASPP, attention gates, and deep supervision.
    """
    def __init__(self, in_channels=3, n_classes=19, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.e1 = conv_block(in_channels, 64, dropout=0.1)
        self.e2 = conv_block(64, 128, dropout=0.1)
        self.e3 = conv_block(128, 256, dropout=0.2)
        self.e4 = conv_block(256, 512, dropout=0.2)

        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(512, 1024)

        # Decoder
        self.d4 = DecoderBlock(1024, 512, 512, use_attention=True)
        self.d3 = DecoderBlock(512, 256, 256, use_attention=True)
        self.d2 = DecoderBlock(256, 128, 128, use_attention=True)
        self.d1 = DecoderBlock(128, 64, 64, use_attention=True)

        # Final output layer
        self.final = FinalHead(64, n_classes)

        # Deep supervision branches
        if self.deep_supervision:
            self.ds2 = nn.Conv2d(128, n_classes, 1)
            self.ds3 = nn.Conv2d(256, n_classes, 1)
            self.ds4 = nn.Conv2d(512, n_classes, 1)

    def forward(self, x):
        # Encoder
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))

        # Bottleneck with ASPP
        b = self.aspp(self.pool(s4))

        # Decoder with skip connections
        d4 = self.d4(b, s4)
        d3 = self.d3(d4, s3)
        d2 = self.d2(d3, s2)
        d1 = self.d1(d2, s1)

        # Final segmentation map
        out = self.final(d1)

        # Deep supervision outputs (optional)
        if self.deep_supervision:
            out2 = F.interpolate(self.ds2(d2), size=out.shape[2:], mode='bilinear', align_corners=False)
            out3 = F.interpolate(self.ds3(d3), size=out.shape[2:], mode='bilinear', align_corners=False)
            out4 = F.interpolate(self.ds4(d4), size=out.shape[2:], mode='bilinear', align_corners=False)
            return (out + 0.3 * out2 + 0.3 * out3 + 0.4 * out4) / 2

        return out

"""
References:
* https://github.com/jfhealthcare/Chexpert
"""

from typing import Sequence
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import create_attn, SqueezeExcite


class PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DA_NetAttention(nn.Module):
    """https://github.com/junfu1115/DANet"""

    def __init__(self, num_channels: int):
        super(DA_NetAttention, self).__init__()
        self.sa = PAM_Module(num_channels)
        self.sc = CAM_Module(num_channels)

    def forward(self, input_tensor):
        sa_conv = self.sa(input_tensor)
        sc_conv = self.sc(input_tensor)
        feat_sum = sa_conv + sc_conv
        return feat_sum


class SAModule(nn.Module):
    """
    Re-implementation of spatial attention module (SAM) described in:
    *Liu et al., Dual Attention Network for Scene Segmentation, cvpr2019
    code reference:
    https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    """

    def __init__(self, num_channels):
        super(SAModule, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_map):
        batch_size, num_channels, height, width = feat_map.size()

        conv1_proj = (self.conv1(feat_map).view(batch_size, -1, width * height).permute(0, 2, 1))
        conv2_proj = self.conv2(feat_map).view(batch_size, -1, width * height)

        relation_map = torch.bmm(conv1_proj, conv2_proj)
        attention = F.softmax(relation_map, dim=-1)

        conv3_proj = self.conv3(feat_map).view(batch_size, -1, width * height)

        feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
        feat_refine = feat_refine.view(batch_size, num_channels, height, width)

        feat_map = self.gamma * feat_refine + feat_map

        return feat_map


class CAModule(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
    *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    code reference:
    https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
    """

    def __init__(self, num_channels, reduc_ratio=2):
        super(CAModule, self).__init__()
        self.num_channels = num_channels
        self.reduc_ratio = reduc_ratio

        self.fc1 = nn.Linear(num_channels, num_channels // reduc_ratio, bias=True)
        self.fc2 = nn.Linear(num_channels // reduc_ratio, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        # attention branch--squeeze operation
        gap_out = feat_map.view(feat_map.size()[0], self.num_channels, -1).mean(dim=2)

        # attention branch--excitation operation
        fc1_out = self.relu(self.fc1(gap_out))
        fc2_out = self.sigmoid(self.fc2(fc1_out))

        # attention operation
        fc2_out = fc2_out.view(fc2_out.size()[0], fc2_out.size()[1], 1, 1)
        feat_map = torch.mul(feat_map, fc2_out)

        return feat_map


class sSE(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        super(sSE, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        # spatial squeeze
        input_tensor = feat_map
        batch_size, channel, a, b = input_tensor.size()
        out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class scSE(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels) -> None:
        super(scSE, self).__init__()
        self.cSE = SqueezeExcite(num_channels)
        self.sSE = sSE(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class AttentionMap(nn.Module):
    def __init__(self, attention_map: str, num_channels: int):
        super(AttentionMap, self).__init__()
        if attention_map == "SAM":
            self.attn = SAModule(num_channels)
        elif attention_map == "DANET":
            self.attn = DA_NetAttention(num_channels)
        elif attention_map == "CAM":
            self.attn = CAModule(num_channels)
        elif attention_map == "CBAM":
            self.attn = create_attn("cbam", num_channels)
        elif attention_map == "LCBAM":
            self.attn = create_attn("lcbam", num_channels)
        elif attention_map == "sCSE":
            self.attn = scSE(num_channels)
        elif attention_map is None:
            self.attn = nn.Identity()
        else:
            raise ValueError("Unknown attention type : {}".format(attention_map))

    def forward(self, feat_map):
        return self.attn(feat_map)

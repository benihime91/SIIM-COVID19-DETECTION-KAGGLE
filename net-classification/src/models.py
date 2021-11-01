from collections import OrderedDict

import torch
import torch.nn.functional as F
from fastai.vision.core import apply_init
from mish_cuda import MishCuda
from pytorch_toolbelt.modules.pooling import GWAP, GlobalAvgPool2d
from timm import create_model
from timm.models import EfficientNetFeatures, ResNet
from timm.models.layers import SelectAdaptivePool2d, adaptive_avgmax_pool2d
from timm.models.resnet import Bottleneck
from torch import nn
from torch.nn.modules.linear import Linear
from yacs.config import CfgNode

from .attn import AttentionMap


class MishCuda(MishCuda):
    """Placeholder for inplace argument"""
    def __init__(self, inplace: bool = True): super(MishCuda, self).__init__()

Mish = MishCuda


# ======================================== POOLING LAYERS  ====================================== #
class PcamPool(nn.Module):
    # Credit: https://github.com/jfhealthcare/Chexpert
    def __init__(self):
        super(PcamPool, self).__init__()

    def forward(self, feat_map, logit_map):
        prob_map   = torch.sigmoid(logit_map)
        weight_map = prob_map / prob_map.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        feat = (feat_map * weight_map).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        return feat


# ========================================= MLP HEADS  =========================================== #
class LinearHead(nn.Module):
    def __init__(self, nf: int, cfg: CfgNode):
        super(LinearHead, self).__init__()
        self.cfg = cfg
        self.nf = nf
        self.nc = cfg.INPUT.NUM_CLASSES

        self.attention_map = AttentionMap(cfg.MODEL.HEAD.ATTENTION_MAP, self.nf)

        if cfg.MODEL.HEAD.POOLING == "gwap":
            self.global_pool = GWAP(self.nf, flatten=True)
        elif cfg.MODEL.HEAD.POOLING == "gap":
            self.global_pool = GlobalAvgPool2d(flatten=True)
        else:
            self.global_pool = SelectAdaptivePool2d(
                1, cfg.MODEL.HEAD.POOLING, flatten=True
            )
            self.nf *= self.global_pool.feat_mult()

        if cfg.MODEL.HEAD.MULTI_DROPOUT.USE:
            print("Multi-Sample Dropout is active")
            self.multi_drop = True
            self.dropout = nn.ModuleList()
            for _ in range(cfg.MODEL.HEAD.MULTI_DROPOUT.NUM):
                self.dropout.append(nn.Dropout(cfg.MODEL.HEAD.DROPOUT))
        else:
            self.multi_drop = False
            self.dropout = nn.Dropout(cfg.MODEL.HEAD.DROPOUT)

        self.fc = nn.Linear(self.nf, self.nc)
        apply_init(self, func=nn.init.kaiming_normal_)

    def forward(self, feature_maps):
        feature_maps = self.attention_map(feature_maps)
        x = self.global_pool(feature_maps)

        if not self.multi_drop:
            x = self.dropout(x)
            output = self.fc(x)

        else:
            for i, dropout in enumerate(self.dropout):
                if i == 0:
                    output = self.fc(dropout(x))
                else:
                    output += self.fc(dropout(x))
            output /= len(self.dropout)

        return {"output": output}


class LinBnHead(nn.Module):
    def __init__(self, nf: int, cfg: CfgNode):
        super(LinBnHead, self).__init__()
        self.cfg = cfg
        self.nf = nf
        self.nc = cfg.INPUT.NUM_CLASSES

        self.attention_map = AttentionMap(cfg.MODEL.HEAD.ATTENTION_MAP, self.nf)

        if cfg.MODEL.HEAD.MULTI_DROPOUT.USE:
            print("Multi-Sample Dropout is active")
            self.multi_drop = True
            self.dropout = nn.ModuleList()
            for _ in range(cfg.MODEL.HEAD.MULTI_DROPOUT.NUM):
                self.dropout.append(nn.Dropout(cfg.MODEL.HEAD.DROPOUT))
        else:
            self.multi_drop = False
            self.dropout = nn.Dropout(cfg.MODEL.HEAD.DROPOUT)

        self.bnorm = nn.BatchNorm1d(self.nf)
        self.dropout = nn.Dropout(cfg.MODEL.HEAD.DROPOUT)
        self.fc = nn.Linear(self.nf, self.nc, bias=False)
        apply_init(self, func=nn.init.kaiming_normal_)

    def forward(self, feature_maps):
        feature_maps = self.attention_map(feature_maps)
        x = self.global_pool(feature_maps)
        x = self.bnorm(x)

        if not self.multi_drop:
            x = self.dropout(x)
            output = self.fc(x)

        else:
            for i, dropout in enumerate(self.dropout):
                if i == 0:
                    output = self.fc(dropout(x))
                else:
                    output += self.fc(dropout(x))
            output /= len(self.dropout)

        return {"output": output}


class FastaiHead(nn.Module):
    def __init__(self, nf: int, cfg: CfgNode):
        super(FastaiHead, self).__init__()
        self.cfg = cfg
        self.nf = nf
        self.nc = cfg.INPUT.NUM_CLASSES

        self.attention_map = AttentionMap(cfg.MODEL.HEAD.ATTENTION_MAP, self.nf)

        if cfg.MODEL.HEAD.POOLING == "gwap":
            self.global_pool = GWAP(self.nf, flatten=True)
        elif cfg.MODEL.HEAD.POOLING == "gap":
            self.global_pool = GlobalAvgPool2d(flatten=True)
        else:
            # fmt: off
            self.global_pool = SelectAdaptivePool2d(1, cfg.MODEL.HEAD.POOLING, flatten=True)
            self.nf *= self.global_pool.feat_mult()
            # fmt: on

        layers = [
            nn.BatchNorm1d(self.nf),
            nn.Dropout(cfg.MODEL.HEAD.DROPOUT / 2),
            nn.Linear(self.nf, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(cfg.MODEL.HEAD.DROPOUT),
            nn.Linear(512, self.nc, bias=False),
        ]
        self.layers = nn.Sequential(*layers)
        apply_init(self.layers, func=nn.init.kaiming_normal_)

    def forward(self, feature_maps):
        feature_maps = self.attention_map(feature_maps)
        x = self.global_pool(feature_maps)
        output = self.layers(x)
        return {"output": output}


class PcamPoolHead(nn.Module):
    # Credit: https://github.com/jfhealthcare/Chexpert
    def __init__(self, nf: int, cfg: CfgNode):
        super(PcamPoolHead, self).__init__()
        self.global_pool = PcamPool()
        self.nf = nf
        self.cfg = cfg
        self.nc = cfg.INPUT.NUM_CLASSES
        self.num_classes = [1] * self.nc
        self._init_classifier()
        self.dropout = nn.Dropout2d(cfg.MODEL.HEAD.DROPOUT)

        if self.cfg.MODEL.HEAD.MULTI_DROPOUT.USE:
            print("Multi-Sample Dropout is active")
            self.multi_drop = True
            self.dropout = nn.ModuleList()
            for _ in range(cfg.MODEL.HEAD.MULTI_DROPOUT.NUM):
                self.dropout.append(nn.Dropout2d(cfg.MODEL.HEAD.DROPOUT))
        else:
            self.multi_drop = False

    def _init_classifier(self):
        for index, num_class in enumerate(self.num_classes):
            ll = "fc_" + str(index)
            layers = [
                AttentionMap(self.cfg.MODEL.HEAD.ATTENTION_MAP, self.nf),
                nn.Conv2d(self.nf, num_class, 1)
            ]
            mn = nn.Sequential(*layers)
            setattr(self, ll, mn)
            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def forward(self, feat_map):
        logits = list()
        for index, _ in enumerate(self.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            logit_map  = classifier(feat_map)
            feat  = self.global_pool(feat_map, logit_map)
            feat  = self.dropout(feat)
            logit = classifier(feat)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)
        return {"output": torch.cat(logits, 1)}


class MultiHead(nn.Module):
    """
    https://www.kaggle.com/ttahara/ranzcr-multi-head-model-training
    """

    def __init__(self, nf: int, cfg: CfgNode):
        super(MultiHead, self).__init__()
        self.nf = feats = nf
        self.nc = cfg.INPUT.NUM_CLASSES

        if cfg.MODEL.HEAD.POOLING == "gwap":
            self.global_pool = GWAP(self.nf, flatten=True)
        elif cfg.MODEL.HEAD.POOLING == "gap":
            self.global_pool = GlobalAvgPool2d(flatten=True)
        else:
            self.global_pool = SelectAdaptivePool2d(1, cfg.MODEL.HEAD.POOLING, True)
            self.nf *= self.global_pool.feat_mult()

        for i in range(self.nc):
            ln = f"head_{i}"
            ll = [
                AttentionMap(cfg.MODEL.HEAD.ATTENTION_MAP, feats),
                self.global_pool,
                nn.Linear(self.nf, self.nf),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.MODEL.HEAD.DROPOUT),
                nn.Linear(self.nf, 1),
            ]
            ll = nn.Sequential(*ll)
            apply_init(ll)
            setattr(self, ln, ll)

    def forward(self, feat_map):
        logits = list()
        for index, _ in enumerate(range(self.nc)):
            classifier = getattr(self, "head_" + str(index))
            logit = classifier(feat_map)
            logits.append(logit)
        return {"output": torch.cat(logits, 1)}


def build_head(cfg: CfgNode, nf: int):
    if cfg.MODEL.HEAD.NAME == "lin":
        head = LinearHead
    elif cfg.MODEL.HEAD.NAME == "linbn":
        head = LinBnHead
    elif cfg.MODEL.HEAD.NAME == "pcam":
        head = PcamPoolHead
    elif cfg.MODEL.HEAD.NAME == "fastai":
        head = FastaiHead
    elif cfg.MODEL.HEAD.NAME == "multi_head":
        head = MultiHead
    return head(nf, cfg)


# ============================= AUXILIARY SEGMENTATION HEAD ====================================== #
class SegmHead(nn.Module):
    def __init__(
        self,
        nf: int,
        n_out: int,
        nc: int = 128,
        add_skip: bool = False,
        act_fn=nn.ReLU(inplace=True),
    ):
        super(SegmHead, self).__init__()
        self.downsample = None

        self.conv1 = nn.Conv2d(nf, nc, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nc)

        self.conv2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nc)

        self.act_fn = act_fn
        self.fc = nn.Conv2d(nc, n_out, 1, bias=False, padding=0)

        self.add_skip = add_skip

        if self.add_skip:
            self.downsample = nn.Sequential(
                nn.Conv2d(nf, nc, kernel_size=1, bias=False), nn.BatchNorm2d(nc)
            )

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.act_fn(out)
        return self.fc(out)


# ======================================= BACKBONE BUILDER ====================================== #
def create_backbone(cfg: CfgNode, **kwargs):
    """Create BackBone from Timm"""
    model_name = cfg.MODEL.BACKBONE.NAME
    backbone = create_model(
        model_name=model_name,
        pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
        drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH,
        in_chans=cfg.INPUT.NUM_CHANNELS,
        **cfg.MODEL.BACKBONE.INIT_ARGS,
        **kwargs,
    )
    return backbone


# ===================================== META ARCHITECTURES ======================================= #
class AuxModel(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(AuxModel, self).__init__()

        kwargs = dict(features_only=True, out_indices=(3, 4))
        if cfg.MODEL.USE_MISH:
            kwargs["act_layer"] = Mish
        self.enet = create_backbone(cfg, **kwargs)

        assert isinstance(self.enet, EfficientNetFeatures)

        self.mask_head = SegmHead(
            self.enet.feature_info.channels()[0],
            n_out=1,
            nc=cfg.MODEL.SEG_HEAD.LIN_FTRS,
            add_skip=cfg.MODEL.SEG_HEAD.RESIDUAL,
            act_fn=nn.ReLU(inplace=True),
        )

        kwargs = dict(act_layer=Mish) if cfg.MODEL.USE_MISH else {}
        net = create_backbone(cfg, **kwargs)
        head_block = nn.Sequential(net.conv_head, net.bn2, net.act2)
        pool_head = build_head(cfg, net.num_features)
        self.cls_head = nn.Sequential(head_block, pool_head)

    def forward(self, input: torch.Tensor):
        features = self.enet(input)
        output = self.cls_head(features[1])
        if self.training:
            mask = self.mask_head(features[0])
            output["mask"] = mask
            return output
        else:
            return output

# ===================================== EXPERIMENTAL RESNET ARCH ================================= #
# ** Suports only mask aux loss
class ExResnetModel(nn.Module):
    def __init__(self, cfg: CfgNode, **kwargs):
        super(ExResnetModel, self).__init__()
        resnet = create_backbone(cfg, **kwargs)
        assert isinstance(resnet, ResNet)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.act1 = resnet.act1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.fc = build_head(cfg, resnet.num_features)

        self.mask_head = SegmHead(
            resnet.feature_info[-2]["num_chs"],
            n_out=1,
            nc=cfg.MODEL.SEG_HEAD.LIN_FTRS,
            add_skip=cfg.MODEL.SEG_HEAD.RESIDUAL,
            act_fn=nn.ReLU(inplace=True),
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        l3 = x = self.layer3(x)
        x = self.layer4(x)
        return l3, x

    def forward(self, input: torch.Tensor):
        features = self.forward_features(input)
        output = self.fc(features[1])

        if self.training:
            mask = self.mask_head(features[0])
            output["mask"] = mask
            return output
        else:
            return output


# builder function
def build_model(cfg: CfgNode):
    kwargs = dict(act_layer=Mish) if cfg.MODEL.USE_MISH else {}
    backbone = create_backbone(cfg, **kwargs)
    
    if cfg.TRAINING.AUX_LOSS and cfg.MODEL.AUX_MODEL.VERSION == "v1":
        model = AuxModel(cfg)

    elif cfg.TRAINING.AUX_LOSS and cfg.MODEL.AUX_MODEL.VERSION == "resnet":
        model = ExResnetModel(cfg)
    
    else:
        head = build_head(cfg, nf=backbone.num_features)
        model = nn.Sequential(OrderedDict(backbone=backbone, head=head))
    return model

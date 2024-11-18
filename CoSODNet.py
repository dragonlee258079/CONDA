import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

from functools import reduce
from operator import add

from learner import HPNLearner

from Decoder import Decoder

from B2_VGG import B2_VGG


class _DASPPConvBranch(nn.Module):
    """
    ConvNet block for building DenseASPP.
    """

    def __init__(self, in_channel, out_channel, inter_channel=None, dilation_rate=1, norm='BN'):
        super().__init__()

        if not inter_channel:
            inter_channel = in_channel // 2

        use_bias = norm == ""
        self.conv1 = Conv2d(
            in_channel,
            inter_channel,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, inter_channel),
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(self.conv1)

        self.conv2 = Conv2d(
            inter_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            dilation=dilation_rate,
            padding=dilation_rate,
            bias=use_bias,
            norm=get_norm(norm, out_channel),
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DASPPBlock(nn.Module):
    def __init__(self, cfg):
        super(DASPPBlock, self).__init__()

        enc_last_channel = cfg.MODEL.ENCODER.CHANNEL[-1]
        adap_channel = cfg.MODEL.DASPP.ADAP_CHANNEL
        self.adap_layer = Conv2d(
            enc_last_channel,
            adap_channel,
            kernel_size=1,
            bias=False,
            norm=get_norm('BN', adap_channel),
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(self.adap_layer)

        dilations = cfg.MODEL.DASPP.DILATIONS
        self.convlayers = len(dilations)

        # must be divisible by 32 because of the group norm
        dil_branch_ch = math.ceil(adap_channel/self.convlayers/32)*32

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.DASPP_Conv_Branches = []
        for idx, dilation in enumerate(dilations):
            this_conv_branch = _DASPPConvBranch(
                adap_channel + idx * dil_branch_ch,
                dil_branch_ch,
                inter_channel=adap_channel // 2,
                dilation_rate=dilation,
                norm="BN",
            )
            self.add_module("conv_brach_{}".format(idx + 1), this_conv_branch)
            self.DASPP_Conv_Branches.append(this_conv_branch)

        self.after_daspp = Conv2d(
            adap_channel*2 + dil_branch_ch*self.convlayers,
            adap_channel,
            kernel_size=1,
            bias=False,
            norm=get_norm("BN", adap_channel),
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(self.after_daspp)

    def forward(self, fea):
        fea = self.adap_layer(fea)

        global_pool_fea = self.global_pool(fea).expand_as(fea)

        out_fea = fea
        for idx, layer in enumerate(self.DASPP_Conv_Branches):
            dil_conv_fea = layer(out_fea)
            out_fea = torch.cat([dil_conv_fea, out_fea], dim=1)

        daspp_fea = torch.cat([global_pool_fea, out_fea], dim=1)
        after_daspp_fea = self.after_daspp(daspp_fea)

        return after_daspp_fea


class CoSODNet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super().__init__()

        self.mode = mode
        self.last_fea_name = cfg.MODEL.ENCODER.NAME[-1]
        self.gr_fea_name = cfg.MODEL.GROUP_ATTENTION.NAME[0]

        self.encoder = B2_VGG()

        self.cost_feat_ids = [18, 20, 22, 25, 27, 29, 30]
        self.decoder_feat_ids = [3, 8, 15]
        nbottlenecks = [2, 2, 3, 3, 3, 2]

        self.daspp_block = DASPPBlock(cfg)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]

        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))

        self.decoder = Decoder(cfg)

        self.pred = nn.Conv2d(128, 1, 1)

    def extract_features(self, img):
        cost_feats = []
        dec_feats = []
        feat = img
        for lid, module in enumerate(self.encoder.features):
            feat = module(feat)
            if lid in self.cost_feat_ids:
                cost_feats.append(feat.clone())
            if lid in self.decoder_feat_ids:
                dec_feats.append(feat)
        return dec_feats, cost_feats

    def forward(self, imgs, co_gts=None):
        N, _, _, _ = imgs.size()

        enc_feats, cost_feats = self.encoder(imgs)

        cost_feats.append(self.daspp_block(cost_feats[-1]))

        cost_out_feas, sals_3, sals_2, \
        inds_3, inds_2, cyc_loss_3, cyc_loss_2 = self.hpn_learner(cost_feats, self.stack_ids, imgs, co_gts)

        fea = self.decoder(enc_feats, cost_out_feas)

        return {
            "final_pred": self.pred(fea),
            "sals_3": sals_3,
            "sals_2": sals_2,
            "inds_3": inds_3,
            "inds_2": inds_2,
            "cyc_loss_3": cyc_loss_3,
            "cyc_loss_2": cyc_loss_2
        }

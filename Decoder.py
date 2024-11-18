import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

from GroupAttention import GroupAttention


class Decoder_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder_Conv, self).__init__()

        self.lateral_conv = Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            bias=False,
            norm=get_norm("BN", out_channel)
        )

        in_channel = out_channel

        self.output_conv = Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm("BN", out_channel),
            activation=F.relu,
        )

        if self.lateral_conv is not None:
            weight_init.c2_xavier_fill(self.lateral_conv)
        weight_init.c2_xavier_fill(self.output_conv)

    def forward(self, enc_fea, dec_fea=None):
        if dec_fea is not None:
            cur_fpn = self.lateral_conv(enc_fea)
            dec_fea = cur_fpn + F.interpolate(dec_fea, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
            dec_fea = self.output_conv(dec_fea)
        else:
            dec_fea = self.output_conv(enc_fea)
        return dec_fea


class Triple_Decoder_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Triple_Decoder_Conv, self).__init__()

        self.lateral_conv = Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            bias=False,
            norm=get_norm("BN", out_channel)
        )

        in_channel = out_channel

        self.output_conv = Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm("BN", out_channel),
            activation=F.relu,
        )

        if self.lateral_conv is not None:
            weight_init.c2_xavier_fill(self.lateral_conv)
        weight_init.c2_xavier_fill(self.output_conv)

    def forward(self, enc_fea, cost_fea, dec_fea=None):
        if dec_fea is not None:
            cur_fpn = self.lateral_conv(enc_fea)
            dec_fea = cur_fpn + cost_fea + F.interpolate(dec_fea, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
            dec_fea = self.output_conv(dec_fea)
        else:
            cur_fpn = self.lateral_conv(enc_fea)
            dec_fea = cur_fpn + cost_fea
            dec_fea = self.output_conv(dec_fea)
        return dec_fea


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        fea_channels = cfg.MODEL.ENCODER.CHANNEL
        hidden_dim = cfg.MODEL.COFORMER_DECODER.HIDDEN_DIM

        for idx, channel in enumerate(fea_channels):
            if idx >= 3:
                dec_conv = Triple_Decoder_Conv(
                    in_channel=channel,
                    out_channel=hidden_dim
                )
                self.add_module("decoder_{}".format(idx + 1), dec_conv)
            else:
                dec_conv = Decoder_Conv(
                    in_channel=channel,
                    out_channel=hidden_dim
                )
                self.add_module("decoder_{}".format(idx + 1), dec_conv)

            if idx != 0:
                group_att = GroupAttention(cfg, hidden_dim)
                self.add_module("group_att_{}".format(idx + 1), group_att)

    def forward(self, features, cost_feas):
        features = features[::-1]
        fea_nums = len(features)
        dec_fea = None
        for idx, enc_fea in enumerate(features):
            if idx <= 1:
                decoder_layer = getattr(self, "decoder_{}".format(fea_nums - idx))
                dec_fea = decoder_layer(
                    enc_fea=enc_fea,
                    cost_fea=cost_feas[idx],
                    dec_fea=dec_fea
                )
                group_att = getattr(self, "group_att_{}".format(fea_nums - idx))
                if dec_fea.shape[2] > 16:
                    dec_fea = group_att(dec_fea, ds=True, scale=16. / dec_fea.shape[2])
                else:
                    dec_fea = group_att(dec_fea)
            else:
                decoder_layer = getattr(self, "decoder_{}".format(fea_nums - idx))
                dec_fea = decoder_layer(
                    enc_fea=enc_fea,
                    dec_fea=dec_fea
                )
                if idx != len(features) - 1:
                    group_att = getattr(self, "group_att_{}".format(fea_nums - idx))
                    if dec_fea.shape[2] > 16:
                        dec_fea = group_att(dec_fea, ds=True, scale=16. / dec_fea.shape[2])
                    else:
                        dec_fea = group_att(dec_fea)
        return dec_fea


class Parallel_Decoder(nn.Module):
    def __init__(self, fea_channels, hidden_dim):
        super(Parallel_Decoder, self).__init__()

        # fea_channels = cfg.MODEL.ENCODER.CHANNEL[:3]
        # hidden_dim = cfg.MODEL.COFORMER_DECODER.HIDDEN_DIM

        for idx, channel in enumerate(fea_channels):
            dec_conv = Decoder_Conv(
                in_channel=channel,
                out_channel=hidden_dim
            )
            self.add_module("decoder_{}".format(idx), dec_conv)

    def forward(self, features, dec_fea):
        # features = features[::-1]
        out_features = []
        for idx, enc_fea in enumerate(features):
            decoder_layer = getattr(self, "decoder_{}".format(idx))
            out_features.append(decoder_layer(
                enc_fea=dec_fea,
                dec_fea=enc_fea
            ))
        return out_features

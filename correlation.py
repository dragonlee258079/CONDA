r""" Provides functions that builds/manipulates correlation tensors """
import torch
from einops import rearrange

import torch.nn.functional as F

import math


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = rearrange(support_feat, "b c h w -> b h w c").contiguous()
            support_feat = support_feat.view(-1, ch)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = rearrange(query_feat, "b c h w -> b h w c").contiguous()
            query_feat = query_feat.view(-1, ch)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.unsqueeze(0), support_feat.transpose(0, 1).unsqueeze(0)).squeeze()
            corr = corr.view(bsz, ha, wa, bsz, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]

    @classmethod
    def multilayer_correlation_single_layer(cls, query_feats, support_feats, scale=False):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            if scale:
                if support_feat.shape[2] > 8:
                    support_feat = F.interpolate(support_feat, size=(8, 8), mode="bilinear", align_corners=True)
            bsz, ch, hb, wb = support_feat.size()
            support_feat = rearrange(support_feat, "b c h w -> b h w c").contiguous()
            support_feat = support_feat.view(-1, ch)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = rearrange(query_feat, "b c h w -> b h w c").contiguous()
            query_feat = query_feat.view(-1, ch)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.unsqueeze(0), support_feat.transpose(0, 1).unsqueeze(0)).squeeze()
            corr = corr.view(bsz, ha, wa, bsz, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr = torch.stack(corrs).transpose(0, 1).contiguous()

        return corr

    @classmethod
    def multilayer_correlation_local_feats(cls, query_feats, support_feats):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, ha, wa = query_feat.size()
            k_l = support_feat.size()[4]
            k = l = int(math.sqrt(k_l))
            support_feat = rearrange(support_feat, "b c (h w) n (k l) -> (b h w) (n k l) c",
                                     h=ha, w=wa, k=k, l=l).contiguous()
            support_feat = support_feat / (support_feat.norm(dim=-1, p=2, keepdim=True) + eps)

            query_feat = rearrange(query_feat, "b c h w -> (b h w) c").contiguous()
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.unsqueeze(1), support_feat.transpose(1, 2)).squeeze()
            corr = corr.view(bsz, ha, wa, bsz, k, l)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr = torch.stack(corrs).transpose(0, 1).contiguous()

        return corr

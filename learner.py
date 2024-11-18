import torch
import torch.nn as nn
import torch.nn.functional as F

from conv4d import CenterPivotConv4d as Conv4d

from einops import rearrange
from correlation import Correlation
from Decoder import Parallel_Decoder

from ssimLoss import SSIM


class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = x.squeeze(-1).squeeze(-1)
        n, c, h, w, m = x.shape
        x = rearrange(x, "n c h w m -> (n m) c h w")
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        x = rearrange(x, "(n m) c h w -> n h w m c", n=n)
        return x


class HPNLearner(nn.Module):
    def __init__(self, inch, radius=4):
        super(HPNLearner, self).__init__()

        self.radius = radius

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.enc_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.enc_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.enc_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        self.delta_prediction_3 = nn.Conv2d(128, 2, (1, 1), bias=False)
        self.delta_prediction_2 = nn.Conv2d(128, 2, (1, 1), bias=False)

        self.sal_prediction_wcost_3 = nn.Conv2d(128, 1, (3, 3), padding=(1, 1), bias=True)
        self.sal_prediction_wcost_2 = nn.Conv2d(128, 1, (3, 3), padding=(1, 1), bias=True)

        self.deform_prediction_3 = nn.Conv2d(128, (2*radius+1)**2*2, (1, 1), bias=False)
        self.deform_prediction_2 = nn.Conv2d(128, (2*radius+1)**2*2, (1, 1), bias=False)

        self.decoder_3 = Parallel_Decoder([128, 128, 128], 512)
        self.decoder_2 = Parallel_Decoder([128, 128, 128], 512)

        self.ssim = SSIM()

        # Mixing building blocks
        self.enc_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.enc_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.dec1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                  nn.ReLU())

        self.dec2 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, n, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 6, 1, 2, 3).contiguous().view(bsz * n * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, n, hb, wb, ch, o_hb, o_wb).permute(0, 4, 5, 6, 1, 2, 3).contiguous()
        return hypercorr

    def get_init_index(self, corr):
        corr = corr.mean(dim=1)
        N, H, W, _, _, _ = corr.shape
        corr = corr.view(N, H, W, N, -1)
        m_v, flatten_index = corr.max(dim=-1)
        spatial_index = torch.cat([(flatten_index // H).unsqueeze(-1), (flatten_index % W).unsqueeze(-1)], dim=-1)
        return spatial_index

    def bilinear_sampler(self, corr, index, mode='bilinear', mask=False):
        n1, c, ha, wa, n2, hb, wb = corr.size()
        corr = rearrange(corr, "n c h w m l d -> (n h w m) c l d")
        # corr = corr.permute(0, 2, 3, 4, 1, 5, 6).contiguous().view(-1, c, hb, wb)
        ygrid, xgrid = index.split([1, 1], dim=-1)

        xgrid = 2*xgrid/(wb-1) - 1
        ygrid = 2*ygrid/(hb-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        corr = F.grid_sample(corr, grid, align_corners=True)
        corr = rearrange(corr, "(n h w m) c l d -> n c h w m l d", n=n1, h=ha, w=wa, c=c)
        return corr

    def local_selection(self, corr, index):
        r = self.radius
        dx = torch.linspace(-r, r, 2*r+1)
        dy = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(corr.device)

        N, H, W, _, _ = index.shape
        centroid_index = index.reshape(N*H*W*N, 1, 1, 2)
        delta_index = delta.view(1, 2*r+1, 2*r+1, 2)
        local_index = centroid_index + delta_index

        local_corr = self.bilinear_sampler(corr, local_index)
        return local_corr

    def deform_selection(self, corr, index, agg_fea, layer):
        N, H, W, _, _ = index.shape
        r = self.radius

        agg_feas = agg_fea.split(dim=-1, split_size=1)
        if layer == 3:
            delta_index = []
            for fea in agg_feas:
                delta_index.append(
                    self.deform_prediction_3(fea.squeeze(-1)).permute(0, 2, 3, 1).view(N, H, W, 1, 2*r+1, 2*r+1, 2)
                )
            delta_index = torch.cat(delta_index, dim=3)
        if layer == 2:
            delta_index = []
            for fea in agg_feas:
                delta_index.append(
                    self.deform_prediction_2(fea.squeeze(-1)).permute(0, 2, 3, 1).view(N, H, W, 1, 2*r+1, 2*r+1, 2)
                )
            delta_index = torch.cat(delta_index, dim=3)

        delta_index = delta_index.view(N*H*W*N, 2*r+1, 2*r+1, 2)
        delta_index[:, r, r, :] = 0

        centroid_index = index.reshape(N*H*W*N, 1, 1, 2)
        # delta_index = delta.view(1, 2*r+1, 2*r+1, 2)
        local_index = centroid_index + delta_index

        local_corr = self.bilinear_sampler(corr, local_index)
        return local_corr

    def get_local_cost(self, corr, agg_fea=None, pre_index=None, delta=None, layer=3):
        if delta is None:
            if pre_index is None:
                index = self.get_init_index(corr)
            else:
                index = pre_index
        else:
            index = pre_index + delta
        if agg_fea is None:
            local_cost = self.local_selection(corr, index)
        else:
            agg_fea = agg_fea.squeeze(-1).squeeze(-1)
            local_cost = self.deform_selection(corr, index, agg_fea, layer)
        return local_cost, index

    def calculate_img_cycle_loss(self, img, index, co_gts, visual=False):
        n, h, w, m, k = index.shape

        img_rep = img.unsqueeze(0).repeat(m, 1, 1, 1, 1)
        img_rep = rearrange(img_rep, "n m c h w -> (n m) c h w")

        yind, xind = index.split([1, 1], dim=-1)
        xind = 2 * xind / (w - 1) - 1
        yind = 2 * yind / (h - 1) - 1
        ind = torch.cat([xind, yind], dim=-1)

        img_forward_warp = F.grid_sample(
            img_rep,
            rearrange(ind, "n h w m k -> (n m) h w k"),
            align_corners=True)

        img_backward_warp = F.grid_sample(
            img_forward_warp,
            rearrange(ind, "n h w m k -> (m n) h w k"),
            align_corners=True)

        img_backward_warp = rearrange(img_backward_warp, "(n m) c h w -> n m c h w", n=n, m=m)

        cycloss = 0
        for i in range(m):
            cycloss += self.ssim(img_backward_warp[i, :, :, :, :], img, co_gts)
        cycloss /= m

        if visual:
            import transforms as trans
            transform = trans.Compose([
                trans.ToPILImage()
            ])
            for i in range(12):
                outputImage = transform(img[i].data.cpu())
                outputImage.save('visual_imgcycloss/img_{}_{}.jpg'.format(i, round(cycloss.item(), 3)))
            for i in range(12):
                for j in range(12):
                    outputImage = transform(img_backward_warp[i, j].data.cpu())
                    outputImage.save('visual_imgcycloss/warp_{}_{}_{}.jpg'.format(i, j, round(cycloss.item(), 3)))

        return cycloss

    def forward_(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.enc_layer4(hypercorr_pyramid[0])

        pre_index_3, delta_3 = None, None
        hypercorr_sqz3 = None
        sals_3 = []
        inds_3 = []
        for i in range(5):
            local_cost_3, pre_index_3 = self.get_local_cost(hypercorr_pyramid[1], hypercorr_sqz3, pre_index_3, delta_3,
                                                            layer=3)
            hypercorr_sqz3 = self.enc_layer3(local_cost_3)
            sals_3.append(self.sal_prediction_wcost_3(hypercorr_sqz3.squeeze(-1).squeeze(-1).mean(-1)))
            delta_3 = self.delta_prediction_3(hypercorr_sqz3)
            inds_3.append(pre_index_3)

        pre_index_2, delta_2 = None, None
        hypercorr_sqz2 = None
        sals_2 = []
        inds_2 = []
        for i in range(5):
            local_cost_2, pre_index_2 = self.get_local_cost(hypercorr_pyramid[2], hypercorr_sqz2, pre_index_2, delta_2,
                                                            layer=2)
            hypercorr_sqz2 = self.enc_layer2(local_cost_2)
            sals_2.append(self.sal_prediction_wcost_2(hypercorr_sqz2.squeeze(-1).squeeze(-1).mean(-1)))
            delta_2 = self.delta_prediction_2(hypercorr_sqz2)
            inds_2.append(pre_index_2)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-5:-3])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.enc_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-5:-3])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.enc_layer3to2(hypercorr_mix432)

        bsz, ch, ha, wa, n, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.dec1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        dec_fea = self.dec2(hypercorr_decoded)

        return dec_fea, sals_3, sals_2, inds_3, inds_2

    def forward(self, cost_feats, stack_ids, imgs, co_gts):
        cost_feats = cost_feats[::-1]
        corr_4 = Correlation.multilayer_correlation_single_layer(
            cost_feats[:stack_ids[0]],
            cost_feats[:stack_ids[0]]
        )
        hypercorr_sqz4 = self.enc_layer4(corr_4)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, cost_feats[stack_ids[0]].size()[-2:])

        cost_feats_split_3 = cost_feats[stack_ids[0]:stack_ids[1]]
        cost_feats_split_3 = self.decoder_3(cost_feats_split_3, hypercorr_sqz4.squeeze(-1).squeeze(-1).mean(-1))
        corr_3 = Correlation.multilayer_correlation_single_layer(
            cost_feats_split_3,
            cost_feats_split_3
        )

        if co_gts is not None:
            imgs_scale3 = F.interpolate(imgs, cost_feats[stack_ids[0]].size()[-2:], mode="bilinear")
            co_gts_scale3 = F.interpolate(co_gts, cost_feats[stack_ids[0]].size()[-2:], mode="bilinear")

        pre_index_3, delta_3 = None, None
        hypercorr_sqz3 = None
        sals_3 = []
        inds_3 = []
        cyc_loss_3 = 0
        for i in range(2):
            local_cost_3, pre_index_3 = self.get_local_cost(corr_3, hypercorr_sqz3, pre_index_3, delta_3,
                                                            layer=3)
            if co_gts is not None:
                # rec_loss_3 += self.calculate_restruction_loss(imgs_scale3, pre_index_3, co_gts_scale3)
                cyc_loss_3 += self.calculate_img_cycle_loss(imgs_scale3, pre_index_3, co_gts_scale3, visual=False)
            hypercorr_sqz3 = self.enc_layer3(local_cost_3)
            sals_3.append(self.sal_prediction_wcost_3(hypercorr_sqz3.squeeze(-1).squeeze(-1).mean(-1)))
            n, c, h, w, m, _, _ = hypercorr_sqz3.shape
            delta_3 = self.delta_prediction_3(rearrange(hypercorr_sqz3.squeeze(-1).squeeze(-1), "n c h w m -> (n m) c h w"))
            delta_3 = rearrange(delta_3, "(n m) c h w -> n h w m c", n=n)
            inds_3.append(pre_index_3)
        cyc_loss_3 /= 2.

        hypercorr_sqz3_ = self.interpolate_support_dims(hypercorr_sqz3, cost_feats[stack_ids[1]].size()[-2:])
        cost_feats_split_2 = cost_feats[stack_ids[1]:stack_ids[2]]
        cost_feats_split_2 = self.decoder_2(cost_feats_split_2, hypercorr_sqz3_.squeeze(-1).squeeze(-1).mean(-1))
        corr_2 = Correlation.multilayer_correlation_single_layer(
            cost_feats_split_2,
            cost_feats_split_2
        )

        if co_gts is not None:
            imgs_scale2 = F.interpolate(imgs, cost_feats[stack_ids[1]].size()[-2:], mode="bilinear")
            co_gts_scale2 = F.interpolate(co_gts, cost_feats[stack_ids[1]].size()[-2:], mode="bilinear")

        pre_index_2, delta_2 = None, None
        hypercorr_sqz2 = None
        sals_2 = []
        inds_2 = []
        cyc_loss_2 = 0
        for i in range(2):
            local_cost_2, pre_index_2 = self.get_local_cost(corr_2, hypercorr_sqz2, pre_index_2, delta_2,
                                                            layer=2)
            if co_gts is not None:
                # rec_loss_2 += self.calculate_restruction_loss(imgs_scale2, pre_index_2, co_gts_scale2)
                cyc_loss_2 += self.calculate_img_cycle_loss(imgs_scale2, pre_index_2, co_gts_scale2, visual=False)
            hypercorr_sqz2 = self.enc_layer2(local_cost_2)
            n, c, h, w, m, _, _ = hypercorr_sqz2.shape
            sals_2.append(self.sal_prediction_wcost_2(hypercorr_sqz2.squeeze(-1).squeeze(-1).mean(-1)))
            delta_2 = self.delta_prediction_2(rearrange(hypercorr_sqz2.squeeze(-1).squeeze(-1), "n c h w m -> (n m) c h w"))
            delta_2 = rearrange(delta_2, "(n m) c h w -> n h w m c", n=n)
            inds_2.append(pre_index_2)
        cyc_loss_2 /= 2.

        hypercorr_encoded = []
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.enc_layer4to3(hypercorr_mix43)
        hypercorr_encoded.append(hypercorr_mix43.squeeze(-1).squeeze(-1).mean(dim=-1))

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-5:-3])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.enc_layer3to2(hypercorr_mix432)
        hypercorr_encoded.append(hypercorr_mix432.squeeze(-1).squeeze(-1).mean(dim=-1))

        return hypercorr_encoded, sals_3, sals_2, inds_3, inds_2, cyc_loss_3, cyc_loss_2

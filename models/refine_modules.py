#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Refinement-specific modules (the contributions of this work).

This file contains the modules that differ from the reference BP-Net framework:
a GNN-based sparse-to-dense pre-aggregation head (``GNNPreAgg``), a geometry
modulated dynamic-convolution fusion block (``LKP`` / ``SKA`` / ``GeoLSConv`` /
``GeoLSBlock`` / ``LSFuseForPMP``), and the ``PMP`` decoder stage that wires them
together with the shared BP-Net building blocks.
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd

from .base_blocks import Basic2d, Conv1x1, Conv3x3, CSPN, WPool, UpCat, Ident

try:
    from .ops_ska import SkaFn
    HAS_TRITON = True
    print("[INFO] Triton SKA loaded successfully.")
except ImportError as e:
    HAS_TRITON = False
    print(f"[WARN] Triton not found or failed ({e}). Fallback to PyTorch unfold (slower).")

# Add the exts directory to the Python search path (for the gnn_knn extension)
exts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exts"))
if exts_path not in sys.path:
    sys.path.insert(0, exts_path)

import gnn_knn

__all__ = [
    'DenseDepthHead',
    'GNNPreAgg',
    'SqueezeExcite',
    'ConvBN',
    'RepDW',
    'GeoFFN',
    'LKP',
    'SKA',
    'GeoLSConv',
    'GeoLSBlock',
    'LSFuseForPMP',
    'PMP',
]


class DenseDepthHead(nn.Module):
    """
    Lightweight CNN that predicts a dense (1-channel) depth map from features fout.
    - The final Softplus keeps depth non-negative and numerically stable.
    """
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            Basic2d(in_ch, mid_ch, norm_layer=nn.BatchNorm2d, act=nn.ReLU, kernel_size=3, padding=1),
            Basic2d(mid_ch, mid_ch, norm_layer=nn.BatchNorm2d, act=nn.ReLU, kernel_size=3, padding=1),
            nn.Conv2d(mid_ch, 1, kernel_size=3, padding=1, bias=True),
        )
        self.out_act = nn.Softplus()

    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, fout):
        return self.out_act(self.net(fout))  # Bx1xHxW


# -----------------------------
# GNN-based PreAgg
# -----------------------------
class GNNPreAgg(nn.Module):
    """
    GNN-based Pre-Agg. For each image:
        1) Use DenseDepthHead to get d_pred, then hard-fill with sparse S to form d0;
        2) Downsample fout / d0 / S to Hc x Wc (controlled by scale, default 4);
        3) Treat the non-zero pixels of S_c as "sparse-point nodes";
        4) For each pixel (every location on the coarse grid) find the nearest k
           sparse points (k <= self.k);
        5) Use a small MLP to compute per-edge weights, apply softmax, aggregate
           (d_sparse - d0_pixel) into a residual, then add it back to d0_pixel;
        6) Upsample back to the original resolution and apply the observation
           protection with S once more.

    Interface:
        forward(fout, S) -> d_out (B x 1 x H x W)
    """
    def __init__(
        self,
        feat_ch: int,
        guide_ch: int = 16,
        k: int = 6,
        scale: int = 4,
        eps: float = 1e-6,
        final_replace_obs: bool = True,
    ):
        super().__init__()
        assert scale >= 1 and (scale & (scale - 1) == 0), "scale must be a power of 2 (1,2,4,8...) for easy downsampling"
        self.k = int(k)
        self.scale = int(scale)
        self.eps = float(eps)
        self.final_replace_obs = bool(final_replace_obs)

        # Initial dense depth estimate
        self.dense_head = DenseDepthHead(in_ch=feat_ch, mid_ch=64)

        # Guide features: extract low-dim features from fout (used to build edge features)
        self.embed = nn.Conv2d(feat_ch, guide_ch, kernel_size=1, bias=True)
        self.guide_ch = guide_ch

        # Edge-feature MLP: input [g_pix, g_node, delta_xy, delta_d] -> scalar logit
        edge_in_dim = guide_ch * 2 + 2 + 1   # g_pix + g_node + (dy,dx) + depth_diff
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    @staticmethod
    def _downsample_bilinear(x, scale: int):
        if scale == 1:
            return x
        B, C, H, W = x.shape
        Hc, Wc = H // scale, W // scale
        return F.interpolate(x, size=(Hc, Wc), mode='bilinear', align_corners=False)

    @staticmethod
    def _downsample_nearest(x, scale: int):
        if scale == 1:
            return x
        B, C, H, W = x.shape
        Hc, Wc = H // scale, W // scale
        return F.interpolate(x, size=(Hc, Wc), mode='nearest')

    @staticmethod
    def _upsample_bilinear(x, size_hw):
        H, W = size_hw
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, fout, S):
        """
        fout: B x C_feat x H x W
        S   : B x 1      x H x W   (sparse depth, 0 means invalid)
        """
        B, C_feat, H, W = fout.shape
        device = fout.device
        # 1) Dense prediction & hard fill
        d_pred = self.dense_head(fout)             # Bx1xHxW
        mask_obs = (S > 1e-3).float()
        d0 = mask_obs * S + (1. - mask_obs) * d_pred  # Bx1xHxW

        # 2) Downsample to the coarse scale
        fout_c = self._downsample_bilinear(fout, self.scale)    # BxCfxHcWc
        d0_c   = self._downsample_bilinear(d0,   self.scale)    # Bx1xHcWc
        S_c    = self._downsample_nearest(S,     self.scale)    # Bx1xHcWc
        Hc, Wc = d0_c.shape[-2:]


        # Guide features (coarse)
        g_c = self.embed(fout_c)                               # BxGxHcWc

        # Process per batch sample
        d_out_list = []
        for b in range(B):
            d_out_b = self._gnn_single(
                g_c[b],              # GxHcWc
                d0_c[b],             # 1xHcWc
                S_c[b],              # 1xHcWc
            )
            # Upsample back to the original resolution
            d_out_b_up = self._upsample_bilinear(d_out_b.unsqueeze(0), (H, W))[0]   # 1xHxW
            d_out_list.append(d_out_b_up)

        d_out = torch.stack(d_out_list, dim=0)  # Bx1xHxW

        # 3) Final observation protection
        if self.final_replace_obs:
            d_out = mask_obs * S + (1. - mask_obs) * d_out

        return d_out

    @torch._dynamo.disable
    def _gnn_single(self, g_c, d0_c, S_c):
        device = g_c.device
        G, Hc, Wc = g_c.shape

        mask = (S_c[0] > 1e-3)
        idx_nodes = mask.nonzero(as_tuple=False)

        # Flatten to (Hc*Wc, G)
        g_flat = g_c.permute(1, 2, 0).contiguous().view(-1, G)
        M = Hc * Wc

        # --- Case 1: no sparse points -> use a dummy node ---
        if idx_nodes.numel() == 0:
            # Create a fake node to keep the KNN / MLP / DDP paths consistent
            node_coords = torch.zeros((1, 2), device=device)     # (1,2)
            node_depths = torch.zeros((1,), device=device)       # (1,)
            node_feats  = torch.zeros((1, G), device=device)     # (1,G)

            Ns = 1
            k_eff = 1

        # --- Case 2: normal case with sparse points ---
        else:
            Ns = idx_nodes.shape[0]
            k_eff = min(self.k, Ns)

            node_coords = idx_nodes.float()                              # (Ns,2)
            node_depths = S_c[0, idx_nodes[:, 0], idx_nodes[:, 1]]       # (Ns,)
            node_lin_idx = idx_nodes[:, 0] * Wc + idx_nodes[:, 1]        # (Ns,)
            node_feats = g_flat[node_lin_idx]                            # (Ns,G)

        # ------------- Pixel coordinates (M,2) ----------------
        yy, xx = torch.meshgrid(
            torch.arange(Hc, device=device),
            torch.arange(Wc, device=device),
            indexing='ij'
        )
        pix_coords = torch.stack([yy, xx], dim=-1).view(-1, 2).float()   # (M,2)

        # ------------- KNN ---------------
        knn_idx = gnn_knn.knn_idx(
            pix_coords.contiguous(),
            node_coords.contiguous(),
            k_eff
        )   # (M, k_eff)

        # ---------- Feature gathering -----------
        g_pix = g_flat                                        # (M,G)
        g_pix_exp = g_pix.unsqueeze(1).expand(-1, k_eff, -1)  # (M,k,G)
        g_node = node_feats[knn_idx]                          # (M,k,G)

        coord_diff = pix_coords.unsqueeze(1) - node_coords[knn_idx]   # (M,k,2)

        d0_flat = d0_c.view(-1)
        d0_pix = d0_flat.unsqueeze(1)

        depth_sparse = node_depths[knn_idx]                   # (M,k)
        depth_diff = depth_sparse - d0_pix                    # (M,k)

        # ---------- Edge feature -----------
        edge_feat = torch.cat(
            [g_pix_exp, g_node, coord_diff, depth_diff.unsqueeze(-1)],
            dim=-1  # -> (M,k,2G+3)
        ).view(M * k_eff, -1)

        # ---------- MLP & softmax ----------
        edge_logits = self.edge_mlp(edge_feat).view(M, k_eff)
        edge_w = F.softmax(edge_logits, dim=1)

        # ---------- Aggregation -----------
        res = (edge_w * depth_diff).sum(dim=1)
        d_out_flat = d0_flat + res

        d_out_c = d_out_flat.view(1, Hc, Wc)
        return d_out_c


# =========================
# Utils & Basic Blocks
# =========================

class SqueezeExcite(nn.Module):
    def __init__(self, ch, se_ratio=0.25):
        super().__init__()
        hid = max(8, int(ch * se_ratio))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, hid, 1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(hid, ch, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class ConvBN(nn.Module):
    """ Conv + BN (+ optional Act); the initial value of BN.weight is configurable """
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1, bn_weight_init=1.0, act=None, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=bias)
        self.bn   = nn.BatchNorm2d(out_ch)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0.0)
        self.act = act
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class RepDW(nn.Module):
    """
    Lightweight mixer: a residual block of Depthwise 3x3 + Pointwise 1x1
    (ConvNeXt style, but keeping BN).
    """
    def __init__(self, ch, expansion=2, ls_init=1e-3):
        super().__init__()
        hid = ch * expansion
        self.dw  = ConvBN(ch, ch, k=3, p=1, g=ch, act=None)
        self.pw1 = ConvBN(ch, hid, k=1, act=nn.SiLU(inplace=True))
        self.pw2 = ConvBN(hid, ch, k=1, act=None)
        # LayerScale: stabilizes early training
        self.gamma_ls = nn.Parameter(ls_init * torch.ones(ch), requires_grad=True)
    def forward(self, x):
        y = self.dw(x)
        y = self.pw1(y)
        y = self.pw2(y)
        return x + y * self.gamma_ls.view(1, -1, 1, 1)


class GeoFFN(nn.Module):
    """ Channel feed-forward: 1x1 -> SiLU -> 1x1 (final BN=0 so the residual starts from identity) """
    def __init__(self, ch, expansion=2):
        super().__init__()
        hid = ch * expansion
        self.pw1 = ConvBN(ch, hid, k=1, act=nn.SiLU(inplace=True))
        self.pw2 = ConvBN(hid, ch, k=1, bn_weight_init=0.0, act=None)
    def forward(self, x):
        return x + self.pw2(self.pw1(x))


# =========================
# LKP & SKA (dynamic depthwise conv)
# =========================
class LKP(nn.Module):
    """
    Modified LKP: supports gene_ratio (grouped weight generation).
    """
    def __init__(self, dim, lks=7, sks=3, gene_ratio=1):  # added gene_ratio argument
        super().__init__()
        assert sks in (3, 5, 7), "sks (small kernel size) should be 3/5/7"

        self.gene_ratio = gene_ratio
        self.gene_dim = dim // gene_ratio  # the number of generated weight channels is reduced

        mid = max(32, dim // 2)
        self.cv1 = ConvBN(dim, mid, k=1, act=nn.ReLU(inplace=True))
        self.dw  = ConvBN(mid, mid, k=lks, p=(lks-1)//2, g=mid, act=nn.ReLU(inplace=True))
        self.cv2 = ConvBN(mid, mid, k=1, act=nn.ReLU(inplace=True))

        # the number of output channels is greatly reduced
        self.cv3 = nn.Conv2d(mid, self.gene_dim * (sks * sks), kernel_size=1, bias=True)
        self.gn  = nn.GroupNorm(num_groups=max(1, self.gene_dim // 8), num_channels=self.gene_dim * (sks * sks))

        self.sks = sks
        self.dim = dim

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.cv3(self.cv2(self.dw(self.cv1(x))))
        y = self.gn(y)
        # Reshape to (B, gene_dim, 9, H, W)
        y = y.view(b, self.gene_dim, self.sks * self.sks, h, w)
        return y

class SKA(nn.Module):
    def __init__(self, sks=3):
        super().__init__()
        self.sks = sks

    def forward(self, x, w):
        """
        x: (B, C, H, W)
        w: (B, C_gene, K*K, H, W)
        """
        # -----------------------------------------------------------
        # If Triton is available (HAS_TRITON=True), the Triton kernel handles
        # broadcasting automatically. The PyTorch fallback below is only used
        # when Triton is unavailable.
        # -----------------------------------------------------------

        # Check whether grouped weights are used
        B, C, H, W = x.shape
        C_gene = w.shape[1]

        if HAS_TRITON and x.is_cuda:
             if not x.is_contiguous(): x = x.contiguous()
             if not w.is_contiguous(): w = w.contiguous()
             return SkaFn.apply(x, w)

        # --- PyTorch fallback (high memory usage, but must stay numerically correct) ---
        k = self.sks
        pad = (k - 1) // 2

        # 1. Unfold input: (B, C*9, HW)
        patches = F.unfold(x, kernel_size=k, padding=pad, stride=1)
        patches = patches.view(B, C, k*k, -1) # (B, C, 9, HW)

        # 2. Handle weights (replicate if grouped)
        dyn = w.view(B, C_gene, k*k, -1)      # (B, C_gene, 9, HW)

        if C > C_gene:
            # Repeat weights to match the input channels
            ratio = C // C_gene
            # (B, C_gene, ...) -> (B, C_gene, ratio, ...) -> (B, C, ...)
            dyn = dyn.unsqueeze(2).expand(-1, -1, ratio, -1, -1).reshape(B, C, k*k, -1)

        # 3. Compute
        out = (patches * dyn).sum(dim=2).view(B, C, H, W)
        return out


# =========================
# Geo-LSConv Block
# =========================

class GeoLSConv(nn.Module):
    """
    LSNet-style geometry-modulated convolution:
    - Project the geometry Pxyz to image channels with 1x1 and add it to x to
      form the kernel input z.
    - LKP predicts per-channel 3x3 dynamic kernels.
    - SKA applies the dynamic convolution to the original x.
    - BN + SE + residual.
    """
    def __init__(self, ch, geo_ch=3, lks=7, sks=3, use_se=True):
        super().__init__()
        self.geo_proj = ConvBN(geo_ch, ch, k=1, act=None)
        self.lkp = LKP(ch, lks=lks, sks=sks)
        self.ska = SKA(sks=sks)
        self.bn  = nn.BatchNorm2d(ch)
        self.se  = SqueezeExcite(ch, 0.25) if use_se else nn.Identity()

    def forward(self, x, g):
        # g: (B, geo_ch, H, W) -- Pxyz recommended
        z = x + self.geo_proj(g)        # generate dynamic kernels from geometry-modulated features
        w = self.lkp(z)                 # (B, C, K*K, H, W)
        y = self.ska(x, w)              # apply dynamic convolution to the original x
        y = self.bn(y)
        y = self.se(y)
        return x + y

class GeoLSBlock(nn.Module):
    """
    Basic block of a stage: RepDW local mixing -> GeoLSConv geometric dynamic kernel -> FFN
    """
    def __init__(self, ch, geo_ch=3, use_se=True):
        super().__init__()
        self.local = RepDW(ch, expansion=2, ls_init=1e-3)
        self.geo   = GeoLSConv(ch, geo_ch=geo_ch, lks=7, sks=3, use_se=use_se)
        self.ffn   = GeoFFN(ch, expansion=2)
    def forward(self, x, g):
        x = self.local(x)
        x = self.geo(x, g)
        x = self.ffn(x)
        return x

# =========================
# LSFuseForPMP
# =========================

class LSFuseForPMP(nn.Module):
    """
    - Keeps the forward(x, d) interface: x = image features, d = geometry
      (Pxyz recommended, B x 3 x H x W).
    - depth: number of stacked blocks.
    """
    def __init__(self, in_ch, dplanes=3, depth=4, use_se=True):
        super().__init__()
        self.proj_in  = ConvBN(in_ch, in_ch, k=1, act=None)
        self.blocks   = nn.ModuleList([GeoLSBlock(in_ch, geo_ch=dplanes, use_se=use_se) for _ in range(depth)])
        self.proj_out = ConvBN(in_ch, in_ch, k=1, act=None)
    def forward(self, x, d):
        y = self.proj_in(x)
        for blk in self.blocks:
            y = blk(y, d)    # d = Pxyz
        y = self.proj_out(y)
        return y


class PMP(nn.Module):
    """
    Pre+MF+Post
    """

    def __init__(self, level, in_ch, out_ch, drop_path, up=True, pool=True, pre_dilation=1):
        super().__init__()
        self.level = level
        if up:
            self.upcat = UpCat(in_ch, out_ch)
        else:
            self.upcat = Ident()
        if pool:
            self.wpool = WPool(out_ch, level=level)
        else:
            self.wpool = Ident()

        self.pre = GNNPreAgg(
            feat_ch=out_ch,
            guide_ch=16,
            k=5,
            scale=1,
            final_replace_obs=False,
        )

        self.fuse = LSFuseForPMP(in_ch=out_ch, dplanes=3, depth=1)
        self.conv = Conv3x3(out_ch, 1, bias=True)
        self.cspn = CSPN(out_ch, pt=3 * (6 - level))

    def pinv(self, S, K, xx, yy):
        fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
        S = S.view(S.shape[0], 1, -1)
        xx = xx.reshape(1, 1, -1)
        yy = yy.reshape(1, 1, -1)
        Px = S * (xx - cx) / fx
        Py = S * (yy - cy) / fy
        Pz = S
        Pxyz = torch.cat([Px, Py, Pz], dim=1).contiguous()
        return Pxyz

    def forward(self, fout, dout, XI, S, K):
        fout = self.upcat(XI, fout, dout)
        Sp = self.wpool(S, fout)
        Kp = K.clone()
        Kp[:, :2] = Kp[:, :2] / 2 ** self.level
        B, _, height, width = Sp.shape
        xx, yy = torch.meshgrid(torch.arange(width, device=Sp.device), torch.arange(height, device=Sp.device),
                                indexing='xy')
        ###############################################################
        # Pre
        dout = self.pre(fout, Sp)  # Bx1xHxW
        ###############################################################
        # MF
        Pxyz = self.pinv(dout, Kp, xx, yy).view(dout.shape[0], 3, dout.shape[2], dout.shape[3])
        fout = self.fuse(fout, Pxyz)
        res = self.conv(fout)
        dout = dout + res
        ###############################################################
        # Post
        dout = self.cspn(fout, dout, Sp)
        return fout, dout

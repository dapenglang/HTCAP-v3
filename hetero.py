import torch
import torch.nn as nn
from dct_utils import dct2, idct2, subband_masks
from prox_tv import tv_one_step
from gating import PixelGate
class HeteroProjection(nn.Module):
    def __init__(self, low_ratio=0.25, mid_ratio=0.5, tv_lambda=0.05):
        super().__init__()
        self.low_ratio = float(low_ratio)
        self.mid_ratio = float(mid_ratio)
        self.tv_lambda = float(tv_lambda)
        self.gate = PixelGate(in_ch=3, bands=3)
    def forward(self, x: torch.Tensor):
        B,C,H,W = x.shape
        X = dct2(x)
        low_m, mid_m, high_m = subband_masks(H, W, self.low_ratio, self.mid_ratio, x.device)
        low_m  = low_m.view(1,1,H,W); mid_m = mid_m.view(1,1,H,W); high_m = high_m.view(1,1,H,W)
        X_low  = X * low_m
        X_mid  = X * mid_m
        X_high = X * high_m
        x_low  = idct2(X_low)
        x_mid  = idct2(X_mid)
        x_high = idct2(X_high)
        x_low_p  = tv_one_step(x_low, lam=self.tv_lambda)
        x_mid_p  = x_mid
        x_high_p = x_high
        stacked = torch.stack([x_low_p, x_mid_p, x_high_p], dim=1)  # B,3,C,H,W
        weights = self.gate(x)                                      # B,3,H,W
        fused = (weights.unsqueeze(2) * stacked).sum(dim=1)         # B,C,H,W
        return fused, weights

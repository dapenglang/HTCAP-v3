import torch
import torch.nn as nn
class Sparsemax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        z, _ = torch.sort(x, descending=True, dim=-1)
        k = torch.arange(1, z.size(1)+1, device=x.device).float().unsqueeze(0)
        z_cumsum = torch.cumsum(z, dim=-1)
        taus = (z_cumsum - 1) / k
        cond = (z > taus).float()
        k_z = cond.sum(dim=-1, keepdim=True).clamp(min=1)
        tau_z = (z_cumsum.gather(1, (k_z-1).long()) - 1) / k_z
        p = torch.clamp(x - tau_z, min=0.0)
        return p.view(orig_shape)
class PixelGate(nn.Module):
    def __init__(self, in_ch=3, bands=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, bands, 1),
        )
        self.sparsemax = Sparsemax()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        probs = self.sparsemax(logits.flatten(2).transpose(1,2))                     .transpose(1,2).view_as(logits)
        return probs

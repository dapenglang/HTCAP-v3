import torch
import torch.nn.functional as F
def tv_one_step(x: torch.Tensor, lam: float = 0.05, eps: float = 1e-3) -> torch.Tensor:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0,1,0,0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0,0,0,1))
    norm = torch.sqrt(dx*dx + dy*dy + eps*eps)
    dxn, dyn = dx / norm, dy / norm
    ddx = dxn[:, :, :, :-1] - F.pad(dxn[:, :, :, :-1], (1,0,0,0))
    ddy = dyn[:, :, :-1, :] - F.pad(dyn[:, :, :-1, :], (0,0,1,0))
    div = F.pad(ddx, (0,1,0,0)) + F.pad(ddy, (0,0,0,1))
    return x - lam * div

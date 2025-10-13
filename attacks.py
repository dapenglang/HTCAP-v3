import torch
import torch.nn.functional as F
def clamp(x, lo=0.0, hi=1.0):
    return x.clamp(lo, hi)
def pgd_attack(model, x, y, eps, alpha, steps, norm="linf"):
    
    x_adv = x.clone().detach()
    if norm == "linf":
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = clamp(x_adv)
    elif norm == "l2":
        noise = torch.randn_like(x_adv)
        noise = noise / (noise.view(noise.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-12)
        x_adv = clamp(x_adv + noise * eps)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        if norm == "linf":
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        elif norm == "l2":
            g = grad / (grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-12)
            x_adv = x_adv + alpha * g
            delta = x_adv - x
            delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-12
            delta = delta * (eps / delta_norm).clamp(max=1.0)
            x_adv = x + delta
        x_adv = clamp(x_adv).detach()
    return x_adv

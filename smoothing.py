import torch, math
def phi_inv(p: float):
    return math.sqrt(2) * torch.erfinv(torch.tensor(2*p-1)).item()
@torch.no_grad()
def estimate_certificate(model, x, y, sigma=0.15, num_samples=1000, batch=100, device="cpu"):
    N = x.size(0)
    num_classes = model.head.out_features
    counts = torch.zeros(N, num_classes, device=device, dtype=torch.long)
    i = 0
    while i < num_samples:
        b = min(batch, num_samples - i)
        noise = torch.randn_like(x) * sigma
        logits = model(x + noise)
        preds = logits.argmax(1)
        for n in range(N):
            counts[n, preds[n]] += 1
        i += b
    probs = counts.float() / num_samples
    pA, idxA = probs.max(dim=1)
    counts[torch.arange(N), idxA] = -1
    pB = counts.float().max(dim=1).values / num_samples
    radius = 0.5 * sigma * (torch.tensor([phi_inv(pa.item()) for pa in pA]) - torch.tensor([phi_inv(pb.item()) for pb in pB]))
    return pA.cpu().tolist(), pB.cpu().tolist(), radius.cpu().tolist()

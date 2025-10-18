import torch
def dct_matrix(N: int, device=None, dtype=None):
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)
    mat = torch.cos((torch.pi / N) * (n + 0.5) * k)
    mat[0, :] *= 1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return mat * torch.sqrt(torch.tensor(2.0 / N, device=device, dtype=dtype))

def dct2(x: torch.Tensor) -> torch.Tensor:
    B,C,H,W = x.shape
    D_H = dct_matrix(H, x.device, x.dtype)
    D_W = dct_matrix(W, x.device, x.dtype)
    x = x.permute(0,1,3,2)
    x = torch.matmul(x, D_H.t())
    x = x.permute(0,1,3,2)
    x = torch.matmul(D_W, x)
    return x

def idct2(X: torch.Tensor) -> torch.Tensor:
    B,C,H,W = X.shape
    D_H = dct_matrix(H, X.device, X.dtype)
    D_W = dct_matrix(W, X.device, X.dtype)
    X = X.permute(0,1,3,2)
    X = torch.matmul(X, D_H)
    X = X.permute(0,1,3,2)
    X = torch.matmul(D_W.t(), X)
    return X

def subband_masks(H, W, low_ratio=0.25, mid_ratio=0.5, device="cpu"):
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    rad = torch.sqrt((yy.float()/H)**2 + (xx.float()/W)**2)
    low  = (rad <= low_ratio).float()
    mid  = ((rad > low_ratio) & (rad <= mid_ratio)).float()
    high = (rad > mid_ratio).float()
    return low, mid, high

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

from hetero import HeteroProjection
from purifier import IdentityPurifier
from attacks import pgd_attack

from datasets import cifar10_loaders, cifar100_loaders, build_transforms

class PurifierWithHead(nn.Module):
    def __init__(self, num_classes=10, image_size=32, low_ratio=0.25, mid_ratio=0.5, tv_lambda=0.05):
        super().__init__()
        self.hetero = HeteroProjection(low_ratio=low_ratio, mid_ratio=mid_ratio, tv_lambda=tv_lambda)
        self.purifier = IdentityPurifier()
        self.head = nn.Linear(3*image_size*image_size, num_classes)

    def forward(self, x):
        x_fused, _ = self.hetero(x)
        x_hat = self.purifier(x_fused)
        return self.head(x_hat.flatten(1))

def get_loaders(dataset='fake', data_root='./data', image_size=32, num_classes=10, batch_size=128, num_workers=2, download=True):
    if dataset.lower() == 'cifar10':
        train, test = cifar10_loaders(root=data_root, batch_size=batch_size, num_workers=num_workers, download=download)
        return train, test, 10, 32
    if dataset.lower() == 'cifar100':
        train, test = cifar100_loaders(root=data_root, batch_size=batch_size, num_workers=num_workers, download=download)
        return train, test, 100, 32
    # fallback: FakeData
    tf = transforms.Compose([transforms.ToTensor()])
    train = FakeData(size=2048, image_size=(3, image_size, image_size), num_classes=num_classes, transform=tf)
    test  = FakeData(size=512,  image_size=(3, image_size, image_size), num_classes=num_classes, transform=tf)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size, shuffle=False), num_classes, image_size

@torch.no_grad()
def eval_clean(model, loader, device="cpu"):
    model.eval()
    tot, cor = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        tot += y.numel()
        cor += (preds==y).sum().item()
    return cor / max(1, tot)

def eval_under_attack(model, loader, norm="linf", eps=8/255, alpha=2/255, steps=10, device="cpu"):
    model.eval()
    tot, cor = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps, norm=norm)
        preds = model(x_adv).argmax(1)
        tot += y.numel()
        cor += (preds==y).sum().item()
    return cor / max(1, tot)

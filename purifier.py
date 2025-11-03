import torch.nn as nn
class IdentityPurifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Identity()
    def forward(self, x):
        return self.net(x)

import torch

#define the localization error function
class LocalizationError(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, z):
        return torch.mean(torch.abs(x.permute(1,0)[0] - y)), torch.mean(torch.abs(x.permute(1,0)[1] - z))
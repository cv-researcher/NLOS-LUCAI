import torch

# define the localization loss function
class LocalizationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, z):
        return torch.mean(torch.add(torch.pow((x.permute(1,0)[0] - y), 2), torch.pow((x.permute(1,0)[1] - z), 2)))
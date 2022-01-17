import torch

# define the fully connected layers
class FC(torch.nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.l1 = torch.nn.Linear(512, 256)
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, 64)
        self.l4 = torch.nn.Linear(64, 2)
        self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25)

    def forward(self,x):
        x = self.prelu(self.l1(x))
        x = self.prelu(self.l2(x))
        x = self.prelu(self.l3(x))
        return self.l4(x)
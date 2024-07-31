import torch
from torch import nn


# While it is possible to build your own model from pytorch functions, it is much easier to make a child instance of the torch.nn.Module class
class ExampleMNISTModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(784, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.l2 = nn.Linear(100, 50)
        self.bn_2 = nn.BatchNorm1d(50)
        self.l3 = nn.Linear(50, 10)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input: torch.Tensor, batch_size: int):
        # Easier to understand for people coming from non-ML backgrounds
        input = input.view(batch_size,-1)
        x = self.l1(input)
        x = self.sigmoid(x)
        x = self.bn_1(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        x = self.bn_2(x)
        x = self.l3(x)
        x = self.sigmoid(x)
        return x
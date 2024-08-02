import torch
from torch import nn


# While it is possible to build your own model from pytorch functions, it is much easier to make a child instance of the torch.nn.Module class
class ExampleMNISTModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(784, 100)
        self.l2 = nn.Linear(100, 50)
        self.l3 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        
    def forward(self, input: torch.Tensor, batch_size: int):
        # Easier to understand for people coming from non-ML backgrounds
        input = input.view(batch_size,-1)
        x = self.l1(input)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        return x


def save_preheated_mnist(model: nn.Module):
    import os
    from pathlib import Path
    file_name = str(Path(os.path.dirname(__file__)) / 'weights' / 'MNIST_no_normalization.pt')
    torch.save(model.state_dict(), file_name)
    

def load_preheated_mnist() -> nn.Module:
    import os
    from pathlib import Path
    file_name = str(Path(os.path.dirname(__file__)) / 'weights' / 'MNIST_no_normalization.pt')
    assert os.path.exists(file_name), "file does not exist"
    
    model = ExampleMNISTModel()
    model.load_state_dict(torch.load(file_name, weights_only=True))
    return model
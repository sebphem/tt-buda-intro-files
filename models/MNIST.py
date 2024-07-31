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
        self.relu = nn.ReLU()
        
    def forward(self, input: torch.Tensor, batch_size: int):
        # Easier to understand for people coming from non-ML backgrounds
        input = input.view(batch_size,-1)
        x = self.l1(input)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        return x

def save_preheated_mnist(model: nn.Module, save_file_int=1):
    assert 0 < save_file_int and save_file_int <= 5, "can only have 5 save files, I don't want to overload your computer"
    import os
    from pathlib import Path
    file_name = str(Path(os.path.dirname(__file__)) / 'weights' / f'{save_file_int}.pt')
    
    model = ExampleMNISTModel()
    torch.save(model.state_dict(), file_name)
    

def load_preheated_mnist(save_file_int=1) -> nn.Module:
    assert 0 < save_file_int and save_file_int <= 5, "can only have 5 save files, I don't want to overload your computer"
    import os
    from pathlib import Path
    file_name = str(Path(os.path.dirname(__file__)) / 'weights' / f'{save_file_int}.pt')
    assert os.path.exists(file_name), "file does not exist"
    
    model = ExampleMNISTModel()
    model.load_state_dict(torch.load(file_name, weights_only=True))
    return model


if __name__ == "__main__":
    # simple sanity check
    model = ExampleMNISTModel()
    save_preheated_mnist(model,save_file_int=1)
    
    # Print model's state_dict
    print("Saved Models's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    import time
    time.sleep(2)
    print('\n\n')
    
    model = load_preheated_mnist(1)
    # Print model's state_dict
    print("Loaded Models's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
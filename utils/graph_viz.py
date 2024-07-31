import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_confusion_matrix_given_model(model: nn.Module):
    # horribly inefficient, but I want to keep the main file as simple as possible
    batch_size = 1000
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, transform=transforms.ToTensor())
    data, target = next(iter(test_loader))
    
    with torch.no_grad():
        # typecast the target so that we get intellisense help
        target : torch.Tensor = target
        output = model(data, batch_size)
        preds = torch.argmax(output)
        
        cm = confusion_matrix(target.cpu().numpy(), preds.cpu().numpy())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
        # return disp
        # all the end user needs to do is just
        disp.plot(cmap=plt.cm.Purples)
        disp.show()
        
def graph_loss(**name_and_loss):
    for name, loss in name_and_loss.items():
        pyplot.plot([i for i, val in enumerate(loss)], loss, label=name)
    pyplot.legend()
    pyplot.xlabel("Epoch Number")
    pyplot.ylabel("Loss")
    # return pyplot
    # all the user needs to do is just
    pyplot.show()
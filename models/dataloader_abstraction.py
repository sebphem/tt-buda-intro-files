from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_training_mnist_data(batch_size = 64):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    return train_loader


def get_test_mnist_data(batch_size = 1000):
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
    return test_loader
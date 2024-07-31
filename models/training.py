import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from MNIST import ExampleMNISTModel, load_preheated_mnist, save_preheated_mnist
from dataloader_abstraction import get_test_mnist_data, get_training_mnist_data

import sys
sys.path.append('../')
from utils.graph_viz import graph_loss, graph_accuracy, make_confusion_matrix_given_model

train_loader = get_training_mnist_data()
test_loader = get_test_mnist_data()
test_data, test_values = next(iter(test_loader))
test_batch_size = test_data.size(0)

learning_rate = 0.01
model = ExampleMNISTModel()
loss_func = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1E-6, momentum=)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)


model.train()
training_accuracy = []
training_loss = []
test_accuracy = []
test_loss = []
epochs = 10
epsilon = 1E-6

start_time = time.perf_counter()
for epoch in range(1,epochs+1):
    print('Epoch #',epoch)
    num_samples = len(train_loader.dataset)
    
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # training
        batch_size = data.size(0)
        optimizer.zero_grad()
        output = model(data, batch_size)
        loss = loss_func(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        # accuracy
        preds = torch.argmax(output, dim=1)
        epoch_accuracy += preds.eq(target).sum().item()
        
    
    epoch_loss /= num_samples
    training_loss.append(epoch_loss)
    
    epoch_accuracy /= num_samples
    training_accuracy.append(epoch_accuracy)
    print('Training Loss: ', epoch_loss, ' Accuracy: ', epoch_accuracy)
    
    model.eval()
    with torch.no_grad():
        output = model(test_data, test_batch_size)
        preds = torch.argmax(output, dim=1)
        test_accuracy.append(preds.eq(test_values).sum().item()/test_batch_size)
    
    print('Test Accuracy: ', test_accuracy[-1])
    
    # epsilon breaking - avoids overfitting
    if len(training_loss) > 1 and abs(training_loss[-1] - training_loss[-2])  < epsilon:
        break

end_time = time.perf_counter()

total_time = end_time - start_time
print(f'Training Took {total_time:.1f} seconds')    

save_preheated_mnist(model=model, save_file_int=5)

graph_loss(**{"Training Loss":training_loss, "Test Loss": test_loss})
graph_accuracy(**{"Training Accuracy":training_accuracy, "Test Accuracy": test_accuracy})



make_confusion_matrix_given_model(model)
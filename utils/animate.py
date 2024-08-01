import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
from pathlib import Path
import torch
import torchvision
from torch import nn


# Function to convert a single-channel tensor to a NumPy array
def tensor_to_np(tensor):
    tensor = tensor.squeeze(0)  # Remove the channel dimension
    tensor = tensor.clamp(0, 1)  # Ensure the tensor values are within [0, 1]
    return (tensor * 255).byte().numpy()  # Scale to [0, 255] and convert to NumPy array


tensor_to_PIL = torchvision.transforms.ToPILImage()
PIL_to_tensor = torchvision.transforms.ToTensor()
loss_func = nn.CrossEntropyLoss()

def animate_generative_mnist(model: nn.Module, input_tensor: torch.Tensor, desired_number: int, delta: float, epochs :int, output_filepath: str):    
    # add in initial and ending frames
    epochs += 10
    
    image = tensor_to_PIL(input_tensor)
    tensor_state = {'cur_tensor': input_tensor}
    # state = {'previous_y': initial_y}
    label = torch.Tensor([desired_number]).long()
    
    input_tensor.requires_grad = True


    x = np.arange(10)
    y = np.zeros(10)

    # Create a figure with two subplots
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the image in the first subplot
    img_ax = ax1.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax1.axis('off')  # Hide the axes
    ax1.set_title('Generated Image')

    # Create a bar graph in the second subplot
    bars = ax2.bar(x, y)
    ax2.set_ylim(0, 5)
    ax2.set_xticks(range(10))
    ax2.set_ylabel('Histogram of the output')


    def animate(frame, tensor_state):
        tensor = tensor_state['cur_tensor']
        # First couple of frames are going to be the initial model
        if frame < 5:
            tensor.requires_grad = False
            output = model(tensor, 1).detach().view(-1).numpy()
        elif frame > epochs-6:
            tensor.requires_grad = False
            output = model(tensor, 1).detach().view(-1).numpy()
        else:
            tensor.requires_grad = True
            output = model(tensor, 1)
            loss = loss_func(output, label)
            loss.backward()
            epsilon = delta * tensor.grad
            tensor.requires_grad = False
            tensor += epsilon
            output = output.detach().view(-1).numpy()
        
        # print('tensor shape: ', tensor.shape)
        image = tensor_to_np(tensor.detach())
        img_ax.set_array(image)
        max_height = max(output)
        ax2.set_ylim(0, max_height)
        for bar, height in zip(bars, output):
            bar.set_height(height)
        # I love pythons immutability
        tensor_state['cur_tensor'] = tensor
        return img_ax, bars

    ani = FuncAnimation(fig, animate, frames=epochs, interval=200, blit=False, fargs=(tensor_state,))

    # Save animation as an MP4
    ani.save(output_filepath, writer='imagemagick', fps=10)

if __name__== "__main__":
    import sys
    sys.path.append('..')
    from models.MNIST_no_normalization import load_preheated_mnist
    from utils.animate import animate_generative_mnist
    import torchvision
    import torch
    import numpy as np
    tensor_to_PIL = torchvision.transforms.ToPILImage()
    PIL_to_tensor = torchvision.transforms.ToTensor()
    model = load_preheated_mnist()
    loss_func = torch.nn.CrossEntropyLoss()
    import os
    from pathlib import Path
    file_path = str(Path(os.path.abspath('')) / "white_image.gif")
    black_image = torch.zeros((1,28,28))
    white_image = torch.ones((1,28,28))
    animate_generative_mnist(model=model, input_tensor=white_image, desired_number=9,delta=0.01, epochs=100, output_filepath=file_path)
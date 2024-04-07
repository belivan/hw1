import torch
import torch.nn as nn
import torch.nn.functional as F


def get_fc(inp_dim, out_dim, non_linear='relu'):  # function to get the fully connected layers
    """
    Mid-level API. It is useful to customize your own for large code repo.  # Mid-level-API is a term used to describe the level of abstraction of the API, it is not too high level and not too low level
    # Example of high-level API is the PyTorch's nn.Module, and an example of low-level API is the PyTorch's torch.nn.functional
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []  # list to store the layers
    layers.append(nn.Linear(inp_dim, out_dim))  # add the fully connected layer
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers


class SimpleCNN(nn.Module):  # class to define the simple CNN model
    """
    Model definition
    """
    def __init__(self, num_classes=10, inp_size=28, c_dim=1):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(c_dim, 32, 5, padding=2)  # convolutional layer with 32 output channels, kernel size 5, and padding 2, c_dim is the number of input channels
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)  # convolutional layer with 64 output channels, kernel size 5, and padding 2
        self.nonlinear = nn.ReLU()  # ReLU activation function
        self.pool1 = nn.AvgPool2d(2, 2)  # average pooling layer with kernel size 2 and stride 2
        self.pool2 = nn.AvgPool2d(2, 2)  # average pooling layer with kernel size 2 and stride 2

        # TODO set the correct dim here
        self.flat_dim = 64 * (inp_size // 4) * (inp_size // 4)  # calculate the flat dimension
        # 64 is the number of output channels from the second convolutional layer, and (inp_size // 4) * (inp_size // 4) is the size of the tensor after the second pooling layer

        # Sequential is another way of chaining the layers.
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 128, 'none'))
        self.fc2 = nn.Sequential(*get_fc(128, num_classes, 'none'))

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification logits in shape of (N, Nc)
        """

        N = x.size(0)  # get the batch size
        x = self.conv1(x)  # new dimensions: (N, 32, H, W)
        x = self.nonlinear(x)  # apply the ReLU activation function
        x = self.pool1(x)  # new dimensions: (N, 32, H/2, W/2)

        x = self.conv2(x)  # new dimensions: (N, 64, H/2, W/2)
        x = self.nonlinear(x)  # apply the ReLU activation function
        x = self.pool2(x)  # new dimensions: (N, 64, H/4, W/4)

        flat_x = x.view(N, self.flat_dim)  # flatten the tensor
        out = self.fc1(flat_x)  # fully connected layer
        out = self.fc2(out)  # fully connected layer
        return out

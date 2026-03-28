from torch import nn
import torch

class MLPQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden=1):
        super(MLPQNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.seq_modules = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq_modules(x)
    
    # customized weight initialization
    def customized_weights_init(m):
        # compute the gain
        gain = nn.init.calculate_gain('relu')
        # init the params using uniform
        if isinstance(m, nn.Linear):
            # init the params using uniform
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0)


class CNNQNet(nn.Module):
    def __init__(self, input_size:tuple, hidden_size:int, output_size:int, num_hidden=1):
        super(CNNQNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()

        self.conv_block = nn.Sequential(
            self.conv1,
            self.activation,
            self.maxpool,
            self.conv2,
            self.activation,
            self.maxpool,
            self.conv3,
            self.activation,
            self.maxpool,
            self.flatten)
        
        # used to compute the output size of the conv block
        with torch.no_grad():
            dummy = torch.zeros(input_size)  # (1, C, H, W)
            flat_size = self.conv_block(dummy).shape[1]

        fc_layers = []
        for _ in range(num_hidden):
            fc_layers.append(nn.Linear(flat_size, hidden_size))
            fc_layers.append(self.activation)
            flat_size = hidden_size

        fc_layers.append(nn.Linear(flat_size, output_size))

        # Combine conv and fc layers into a single sequential module
        self.fc_block = nn.Sequential(
            *fc_layers
        )
        

    def forward(self, x):
        # Compute the Convolution Block with FC Block
        y = self.conv_block(x)
        y = self.fc_block(y)
        return y
    
    # customized weight initialization
    def customized_weights_init(m):
        # compute the gain
        gain = nn.init.calculate_gain('relu')
        
        # init the convolutional layer
        if isinstance(m, nn.Conv2d):
            # init the params using uniform
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0)
        
        # init the linear layer
        if isinstance(m, nn.Linear):
            # init the params using uniform
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0)

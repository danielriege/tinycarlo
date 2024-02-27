from tinygrad.tensor import Tensor
import tinygrad.nn as nn

state_dim = (3, 66, 200)
action_dim = 2

class ConvBlock:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(x).relu()
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class LinearBlock:
    def __init__(self, in_features, out_features):
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x).relu()
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class PilotNetActor:
    """
    inspired by the NVIDIA PilotNet architecture
    https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    """

    def __init__(self):
        self.bn = nn.BatchNorm2d(3)
        self.conv_layers = [
            ConvBlock(3, 24, 5, 2),
            ConvBlock(24, 36, 5, 2),
            ConvBlock(36, 48, 5, 2),
            ConvBlock(48, 64, 3, 1),
            ConvBlock(64, 64, 3, 1)
        ]
        self.fc = [
            LinearBlock(64 * 18, 100),
            LinearBlock(100, 50),
            LinearBlock(50, 10),
        ]
        self.output_layer = nn.Linear(10, 2)

    def forward(self, x):
        out = self.bn(x)
        out = out.sequential(self.conv_layers).flatten()
        return self.output_layer(out.sequential(self.fc)).tanh()
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class PilotNetCritic:
    """
    Custom Critic network with same convolutional layer structure as the actor network
    """
    def __init__(self):
        self.bn = nn.BatchNorm2d(3)
        self.conv_layers = [
            ConvBlock(3, 24, 5, 2),
            ConvBlock(24, 36, 5, 2),
            ConvBlock(36, 48, 5, 2),
            ConvBlock(48, 64, 3, 1),
            ConvBlock(64, 64, 3, 1)
        ]
        self.fc_state = [
            LinearBlock(64 * 18, 100),
            LinearBlock(100, 32)
        ]
        self.fc_action = [
            LinearBlock(2, 32)
        ]
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, state, action):
        state = self.bn(state)
        state = state.sequential(self.conv_layers).flatten()
        state = state.sequential(self.fc_state)
        action = action.sequential(self.fc_action)
        out = state.cat(action, dim=1)
        out = self.fc1(out).relu()
        return self.fc2(out)
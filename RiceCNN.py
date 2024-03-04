import torch
import torch.nn as nn
import torch.nn.functional as F

class RiceCNN(nn.Module):
    def __init__(self):
        super(RiceCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                              out_channels=64,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2,
                                 padding=0)
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # 64 chans, 7x7 image
        self.fc2 = nn.Linear(128, 10) # num output classes

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Reshape the tensor the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = RiceCNN()
print(model)


input_data = torch.randn(2, 1, 28, 28)

output = model(input_data)

print("Output shape:", output.shape)
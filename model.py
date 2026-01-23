import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class SimpleCNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)   # (B, 16, 32, 32)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)  # (B, 32, 16, 16)
        self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1)  # (B, 64, 8, 8)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Adaptive pooling to support arbitrary input spatial sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer (after adaptive pooling the feature map is 64 x 1 x 1)
        self.fc = nn.Linear(hidden_dim*4 * 1 * 1, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.adaptive_pool(x) # Global Average Pooling
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = SimpleCNN()
    summary(model, input_size=(16, 3, 512, 512))
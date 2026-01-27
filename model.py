import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class SEBlock(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SimpleCNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, attention=False):
        super(SimpleCNN, self).__init__()
         # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  
        )

        # Block 2
        if(attention):
            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim*2),
                nn.ReLU(inplace=True),
                SEBlock(hidden_dim*2),
                nn.MaxPool2d(2)  
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim*2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  
            )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.fc = nn.Linear(hidden_dim*4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x  

if __name__ == '__main__':
    model = SimpleCNN(attention=True)
    summary(model, input_size=(16, 3, 512, 512))
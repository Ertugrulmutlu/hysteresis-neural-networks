# src/model.py
import torch.nn as nn

def norm_layer(norm_type, channels, groups=8):
    if norm_type == "group":
        return nn.GroupNorm(groups, channels)
    elif norm_type == "layer":
        return nn.LayerNorm([channels, 1, 1])
    else:
        return nn.Identity()

class SimpleCNN(nn.Module):
    def __init__(self, norm="none", gn_groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.norm1 = norm_layer(norm, 32, gn_groups)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = norm_layer(norm, 64, gn_groups)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.norm1(self.conv1(x))))
        x = self.pool(self.relu(self.norm2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

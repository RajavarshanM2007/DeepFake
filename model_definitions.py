# model_definitions.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FuNetA(nn.Module):
    def __init__(self, num_classes=2):
        super(FuNetA, self).__init__()
        # A simple CNN-based architecture, no torchvision, no graph conv
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # assume input image resized to 64x64, after two poolings becomes 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x, graph=None):
        # graph argument kept for compatibility; not used in this version
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

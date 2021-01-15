import torch.nn as nn

class Conv4(nn.Module):

  def __init__(self):
    super(Conv4, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv3 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv4 = nn.Conv2d(64, 64, 3, padding = 1)
    self.bn1 =  nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(64)
    self.bn3 = nn.BatchNorm2d(64)
    self.bn4 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(2)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()

     
  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.maxpool(x)
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.maxpool(x)
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.maxpool(x)
    x = self.relu(self.bn4(self.conv4(x)))
    x = self.maxpool(x)
   
    return self.flatten(x)
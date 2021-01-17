import torch.nn as nn

class Conv6(nn.Module):

  def __init__(self):
    super(Conv6, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv3 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv4 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv5 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv6 = nn.Conv2d(64, 64, 3, padding = 1)
    self.bn1 =  nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(64)
    self.bn3 = nn.BatchNorm2d(64)
    self.bn4 = nn.BatchNorm2d(64)
    self.bn5 = nn.BatchNorm2d(64)
    self.bn6 = nn.BatchNorm2d(64)
    self.dropout1 = nn.Dropout()
    self.dropout2 = nn.Dropout()
    self.dropout3 = nn.Dropout()
    self.dropout4 = nn.Dropout()
    self.dropout5 = nn.Dropout()
    self.dropout6 = nn.Dropout()
    self.maxpool = nn.MaxPool2d(2)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(64,128)
     
  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.maxpool(x)
    x = self.dropout1(x)
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.maxpool(x)
    x = self.dropout2(x)
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.maxpool(x)
    x = self.dropout3(x)
    x = self.relu(self.bn4(self.conv4(x)))
    x = self.maxpool(x)
    x = self.dropout4(x)

    x = self.relu(self.bn5(self.conv5(x)))
    x = self.maxpool(x)
    x = self.dropout5(x)

    x = self.relu(self.bn6(self.conv6(x)))
    x = self.maxpool(x)
    x = self.dropout6(x)
   
    return self.fc(self.flatten(x))
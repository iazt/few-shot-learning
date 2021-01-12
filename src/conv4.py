import torch.nn as nn

class BaselineBB(nn.Module):

  def __init__(self, nclasses):
    super(BaselineBB, self).__init__()

    self.nclasses = nclasses
    self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv3 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv4 = nn.Conv2d(64, 64, 3, padding = 1)
    self.bn1 =  nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(64)
    self.bn3 = nn.BatchNorm2d(64)
    self.bn4 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(2,2)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(1600, self.nclasses)
    self.softmax = nn.Softmax()
    self.flatten = nn.Flatten()
    self.dropout1 = nn.Dropout()    
    self.dropout2 = nn.Dropout()    
    self.dropout3 = nn.Dropout()    
    self.dropout4 = nn.Dropout()

     
  def forward(self, x):
    x = self.bn1(self.relu(self.conv1(x)))
    x = self.maxpool(x)
    x = self.dropout1(x)    
    x = self.bn2(self.relu(self.conv2(x)))
    x = self.maxpool(x)
    x = self.dropout2(x)    
    x = self.bn3(self.relu(self.conv3(x)))
    x = self.maxpool(x)
    x = self.dropout3(x)    
    x = self.bn4(self.relu(self.conv4(x)))
    x = self.maxpool(x)
    x = self.dropout4(x)    
    x = self.fc1(self.flatten(x))
    return x
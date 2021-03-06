import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import WeightNorm

class Linear_classifier(nn.Module):
  def __init__(self, in_features, out_features):
    super(Linear_classifier, self).__init__()
    self.fc = nn.Linear(in_features, out_features)
    self.relu = nn.ReLU()
    #self.softmax =nn.Softmax(dim=1)
  def forward(self, x):
    #return self.softmax(self.fc(x))
    
    return self.fc(self.relu(x))


class Baseline(nn.Module):
    def __init__(self, nclasses, backbone):
      super(Baseline, self).__init__()
      self.features = backbone()
      self.classifier = Linear_classifier(128, nclasses)
      self.nclasses = nclasses

    def forward(self,x):
      out  = self.features.forward(x)
      scores  = self.classifier(out)
      return scores

    def forward_embedding(self,x):
      return self.features.forward(x)

    def forward_classifier(self,x):
        return self.classifier.forward(x)


class CosineLayer(nn.Module):
  def __init__(self, in_features, out_features):
    super(CosineLayer, self).__init__()
    self.fc = nn.Linear(in_features, out_features, bias = False)
    WeightNorm.apply(self.fc, 'weight', dim=0)
    self.scale_factor = 2
    #self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 1e-4)
    cos_dist = self.fc(x_normalized) 
    scores = self.scale_factor*(cos_dist) 
    return scores


class Baseline_plus(nn.Module):
  def __init__(self, nclasses, backbone):
    super(Baseline_plus, self).__init__()
    self.features = backbone()
    self.classifier = CosineLayer(128, nclasses)
    self.nclasses = nclasses

  def forward(self,x):
    out  = self.features.forward(x)
    scores  = self.classifier.forward(out)
    return scores

  def forward_embedding(self,x):
    return self.features.forward(x)

  def forward_classifier(self,x):
    return self.classifier.forward(x)
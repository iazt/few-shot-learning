import torch
import torch.nn as nn
from torch.utils.data import Dataset

def map_labels(labels):
  """ Convert labels to a range between 0 and 4"""
  
  def get_idxs(n):
    b = (labels == n)
    return(b)

  result = map(get_idxs, labels.unique())
  new_labels = torch.zeros_like(labels)
  for i, item in enumerate(result): 
    new_labels += item*(i)
  return new_labels

def freeze_embedding(net):
  for parameter in net.features.parameters():
    parameter.requires_grad = False


class evalDataset(Dataset):
  def __init__(self, images, labels):
    self.images = images
    self.labels = map_labels(torch.tensor(labels))

  def __len__(self):
    return self.images.shape[0]
 
  def __getitem__(self, index):
    return self.labels[index], self.images[index]
 

def train_few_shot(net, n_epochs, support_loader, query_loader):
  freeze_embedding(net)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

  total_support = len(support_loader)*support_loader.batch_size
  total_query = len(query_loader)*query_loader.batch_size
  
  support_features = []
  support_labels = []
  query_features = []
  query_labels = []

  #Get
  net.eval()
  for i, data in enumerate(support_loader, 0):
    with torch.no_grad():
      labels = data[0].cuda()
      inputs = data[1].float().cuda()
      feat = net.forward_features(inputs)
      support_features.append(feat)
      support_labels.append(labels)

  for i, data in enumerate(query_loader, 0):
    with torch.no_grad():
      labels = data[0].cuda()
      inputs = data[1].float().cuda()
      feat = net.forward_features(inputs)
      query_features.append(feat)
      query_labels.append(labels)


  #Few shot train
  net.train()
  for epoch in range(n_epochs):   
    running_loss, running_acc = 0.0, 0.0
    for label, features in zip(support_labels, support_features): 
      optimizer.zero_grad()
      outputs = net.forward_classifier(features)
      loss = criterion(outputs, label)
      loss.backward()
      optimizer.step()
      
  #Few shot eval
  net.eval()
  running_acc = 0.0
  valid_loss = 0.0
  
  for label, features in zip(query_labels, query_features):    
    with torch.no_grad():
      Y_pred = net.forward_classifier(features)
    max_prob, max_idx = torch.max(Y_pred.data, dim=1)
    running_acc += torch.sum(max_idx == label).item()
    loss = criterion(Y_pred, label)
    # record validation loss
    valid_loss+= loss.item()

  return running_acc/total_query, valid_loss/total_query
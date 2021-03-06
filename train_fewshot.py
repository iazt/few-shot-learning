import torch.nn as nn
from src.few_shot_episode import train_few_shot, evalDataset
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def few_shot(net, test_set, n_episodes = 600):
  classes  = [i for i in range(20)]
  k = 5
  N = 5
  idxs = [i for i in range(600)] 
  n_iter = 100
  acc = []
  loss = []
  labels = torch.tensor(test_set.labels)
  images = test_set.images

  
  
  for i in range(n_episodes):
    choosen_classes = np.random.choice(classes, N, replace = False)
    choosen_idxs = np.random.choice(idxs, 21, replace = False)
    support_idxs = choosen_idxs[:k]  #indices de los ejemplos en el dataset de test
    support_idxs = np.array([support_idxs + clas*600 for clas in choosen_classes]).reshape(-1)
    query_idxs = choosen_idxs[k:]
    query_idxs = np.array([query_idxs+ clas*600 for clas in choosen_classes]).reshape(-1)
   
    
    support_set_im = images[support_idxs]
    support_set_labels = labels[support_idxs]
    
    query_set_im = images[query_idxs]
    query_set_labels = labels[query_idxs]
    
    support_dataset = evalDataset(support_set_im, support_set_labels)
    query_dataset = evalDataset(query_set_im, query_set_labels)
    support_loader = DataLoader(support_dataset, batch_size = 4, shuffle=True, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size = 16, shuffle=False, num_workers=4, pin_memory=True)
    current_acc, current_loss = train_few_shot(net, n_iter, support_loader, query_loader)
    acc.append(current_acc)
    loss.append(current_loss)
    info = f'Loss:{current_loss:02.5f}, '
    info += f'Train Acc:{current_acc*100:02.1f}%\n'
    sys.stdout.write(info)
  return acc, loss

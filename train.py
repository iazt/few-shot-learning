import sys
import time
import numpy as np
import pandas as pd
import torch

def train(net, n_epochs, train_loader, val_loader, name, dir_checkpoint = '/content/gdrive/My Drive/fewshot/baseline_checkpoints/'):

  total_train = len(train_loader)*train_loader.batch_size
  total_val = len(val_loader)*val_loader.batch_size
  val_acc = []
  val_loss = []
  train_acc = []
  train_loss = []

  best_acc = 0
  t0 = time.time()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
  for epoch in range(n_epochs):
    net.train()
    running_loss, running_acc = 0.0, 0.0
    for i, data in enumerate(train_loader, 0): # Obtener batch
      labels = data[0].cuda()
      inputs = data[1].float().cuda()
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      # Calcular accuracy sobre el conjunto de validación y almacenarlo
      # para hacer un plot después


      items = (i+1) * train_loader.batch_size
      running_loss += loss.item()
      max_prob, max_idx = torch.max(outputs.data, dim=1)
      running_acc += torch.sum(max_idx == labels).item()
      info = f'\rEpoch:{epoch+1}({items}/{total_train}), '
      info += f'Loss:{running_loss/items:02.5f}, '
      info += f'Train Acc:{running_acc/items*100:02.1f}%'
      sys.stdout.write(info)

    train_acc.append(running_acc/total_train)
    train_loss.append(running_loss/total_train)

    net.eval()
    running_acc = 0.0
    valid_loss = 0.0
    
    for i, data in enumerate(val_loader, 0):
      labels = data[0].cuda()
      inputs = data[1].float().cuda()
      with torch.no_grad():
        Y_pred = net(inputs)
      max_prob, max_idx = torch.max(Y_pred.data, dim=1)
      running_acc += torch.sum(max_idx == labels).item()
      loss = criterion(Y_pred, labels)
      # record validation loss
      valid_loss+= loss.item()


    val_acc.append(running_acc/total_val*100)  
    val_loss.append(valid_loss/total_val)  
    info = f', Val Acc:{running_acc/total_val*100:02.2f}%.\n'
    sys.stdout.write(info)
    if running_acc/total_val > best_acc:
      best_acc = running_acc/total_val
      torch.save(net.state_dict(), dir_checkpoint + name + '_best_model.pth')
      with open('/content/gdrive/My Drive/fewshot/baseline_checkpoints/'+ name +'_epoca.txt', 'w') as f:
        f.write(str(epoch + 1 ))
      

  t1 = time.time()
  df = pd.DataFrame({"val_acc":val_acc, "val_loss":val_loss, "train_acc":train_acc, "train_loss":train_loss})
  df.to_csv('/content/gdrive/My Drive/fewshot/baseline_checkpoints/'+name+'_metrics.csv', index=False)
  print(t1-t0, 'segundos')
  return val_acc, val_loss, t1-t0
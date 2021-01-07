from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle


def unpickle(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data


class miniImagenet(Dataset):
  def __init__(self, path):
    dictionary = unpickle(path)
    self.images = dictionary['image_data']
    self.images = (torch.Tensor(self.images)*2)/255 -1    #reescalado entre [-1, 1]
    self.images = torch.transpose(self.images, 2, 3)
    self.images = torch.transpose(self.images, 2, 1)
    self.labels = [i//600 for i in range(self.__len__())]

  def __len__(self):
    return self.images.shape[0]
 
  def __getitem__(self, index):
    return self.labels[index], self.images[index]
 
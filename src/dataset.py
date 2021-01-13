from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ColorJitter


def unpickle(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data


class miniImagenet(Dataset):
  def __init__(self, path, augmentation = True):
    dictionary = unpickle(path)
    self.augmentation = augmentation
    self.images = dictionary['image_data']
    self.images = (torch.Tensor(self.images)*2)/255 -1    #reescalado entre [-1, 1]
    self.images = torch.transpose(self.images, 2, 3)
    self.images = torch.transpose(self.images, 2, 1)
    self.labels = [i//600 for i in range(self.__len__())]
    self.transform = Compose([RandomCrop(70),
    						  RandomHorizontalFlip(),
    						  ColorJitter(0.05, 0.05, 0.05, 0.05)])
    if self.augmentation:
    	self.images = self.transform(self.images)

  def __len__(self):
    return self.images.shape[0]
 
  def __getitem__(self, index):
    return self.labels[index], self.images[index]
 
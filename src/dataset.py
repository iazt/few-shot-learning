from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

def unpickle(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data


class miniImagenet(Dataset):
  def __init__(self, path, augmentation = True, split = True):
    dictionary = unpickle(path)
    self.split = split
    self.augmentation = augmentation
    self.images = dictionary['image_data']
    self.images = (torch.Tensor(self.images)*2)/255 -1    #reescalado entre [-1, 1]
    self.images = torch.transpose(self.images, 2, 3)
    self.images = torch.transpose(self.images, 2, 1)
    self.labels = [i//600 for i in range(self.__len__())]
    self.transform = Compose([RandomResizedCrop(84),
                  RandomHorizontalFlip(),
                  ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)])

    if split:
      self.images, val_images, self.labels, val_labels = train_test_split(self.images, self.labels, test_size=0.2, random_state=42)
      self.val_dataset = TensorDataset(torch.tensor(val_labels, dtype=torch.long), val_images)

    if self.augmentation:
      l = []
      l.append(self.transform(self.images[:10000]))
      l.append(self.transform(self.images[10000:20000]))
      l.append(self.transform(self.images[20000:]))
      self.images = torch.cat(l, dim = 0)

  def get_val_dataset(self):
    if self.split:
      return self.val_dataset
    return None

  def __len__(self):
    return self.images.shape[0]
 
  def __getitem__(self, index):

    return self.labels[index], self.images[index]
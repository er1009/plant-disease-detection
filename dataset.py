import torch
from torch.utils.data import Dataset
import cv2


labels_dict = {'Septoria': torch.tensor([1,0,0,0,0,0,0], dtype=torch.float32),
         'Powdery Mildew': torch.tensor([0,1,0,0,0,0,0], dtype=torch.float32),
         'Healthy': torch.tensor([0,0,1,0,0,0,0], dtype=torch.float32),
         'Tobacco Mosiac Virus': torch.tensor([0,0,0,1,0,0,0], dtype=torch.float32),
         'Spider Mites': torch.tensor([0,0,0,0,1,0,0], dtype=torch.float32),
         'Calcium Deficiency': torch.tensor([0,0,0,0,0,1,0], dtype=torch.float32),
         'Magnesium Deficiency': torch.tensor([0,0,0,0,0,0,1], dtype=torch.float32)}


class PlantDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]
        label = labels_dict[label]
        label_num = torch.argmax(label)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label, label_num


class LeafDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]
        if label == 1:
          label = torch.tensor([1], dtype=torch.float32)
        else:
          label = torch.tensor([0], dtype=torch.float32)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


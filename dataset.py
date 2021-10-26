import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms

class IMDBWikiDataset(Dataset):
    """Dataset for IMDB Wiki"""
    def __init__(self,csv_file,transform=None):
        self.dframe = pd.read_csv(csv_file,sep = ',')
        self.transform=transform

        self.dframe = self.dframe[self.dframe['age'] >= 1]
        self.dframe = self.dframe[self.dframe['age'] <= 120]

    def __len__(self):
        return len(self.dframe)
    
    def __getitem__(self,idx):
        dframe = self.dframe
        transform = self.transform
        path = dframe.iloc[idx,2]
        image = Image.open(os.path.join(path)).convert('RGB')
        target = dframe.iloc[idx,0]
        #print(target)

        if self.transform:
            image = self.transform(image)
        return image , target
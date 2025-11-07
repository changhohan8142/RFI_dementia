import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfms
from PIL import Image

class DementiaDetectionDataset(Dataset):
    def __init__(self, kind, img_sz):
        df = pd.read_csv('dementia_detection.csv', low_memory=False)
        self.df = df[df[kind]==True][['pngfilename', 'label']]
        norm = tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if kind == 'train':
            self.tfms = tfms.Compose([
                tfms.RandomResizedCrop((img_sz, img_sz)),
                tfms.AutoAugment(),
                tfms.RandomHorizontalFlip(),
                tfms.ToTensor(),
                norm,
            ])
        else:
            self.tfms = tfms.Compose([
                tfms.Resize((img_sz, img_sz)),
                tfms.ToTensor(),
                norm
            ])
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = self.tfms(
            Image.open(r.pngfilename).convert("RGB")
        )
        label = torch.tensor(r.label, dtype=torch.float32)
        return img, label


class DementiaPredictionDataset(Dataset):
    def __init__(self, img_sz):
        df = pd.read_csv('dementia_prediction.csv', low_memory=False)
        self.df = df[['pngfilename', 'event', 'obs_time']]
        
        self.tfms = tfms.Compose([
            tfms.Resize((img_sz, img_sz)),
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = self.tfms(
            Image.open(r.pngfilename).convert("RGB")
        )
        event = torch.tensor(r.event, dtype=torch.float32)
        dur = torch.tensor(r.obs_time, dtype=torch.int32)
        return img, event, dur

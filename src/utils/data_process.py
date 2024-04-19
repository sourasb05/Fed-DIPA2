import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np
import pandas as pd
import random
from tqdm import trange
import json
from sklearn.model_selection import train_test_split
import random
import time

class FeatureDataset(Dataset):
    def __init__(self, features_folder, annotations_file):
        self.features_folder = features_folder
        self.annotations = pd.read_csv(annotations_file, header=None, names=['filename', 'label'])
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        filename, label = self.annotations.iloc[idx]
        feature_path = os.path.join(self.features_folder, filename)
        feature = torch.load(feature_path)
        return feature, label




class FeatureDataset_mm(Dataset):
    def __init__(self, features_folder, annotations_file, context_columns):
        self.features_folder = features_folder
        self.annotations = pd.read_csv(annotations_file)
        self.context_columns = context_columns  # Names of the columns with contextual info
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get image features
        filename = self.annotations.iloc[idx]['ImgID']
        feature_path = os.path.join(self.features_folder, filename)
        feature = torch.load(feature_path)

        # Get contextual information and label
        context = self.annotations.iloc[idx][self.context_columns].values.astype(float)
        context = torch.tensor(context, dtype=torch.float32)

        # Get label (event memory score)
        label = self.annotations.iloc[idx]['Mem_s']
        label = torch.tensor(label, dtype=torch.float32)
        
        return feature, context, label
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import json
import sys

from torchvision.ops import roi_pool, roi_align
import torchvision
import torch
import torch.nn as nn

class ImageMaskDataset(Dataset):
    def __init__(self, mega_table, image_folder, input_vector, image_size, flip = False, save_mask = False, flip_prob = 0.5, device='cpu'):
        self.mega_table = mega_table
        self.category_num = len(mega_table['category'].unique())
        self.input_dim = len(input_vector)
        self.image_folder = image_folder
        self.input_vector = input_vector
        self.image_size = image_size
        self.flip_prob = flip_prob
        self.save_mask = save_mask
        self.flip = flip
        self.padding_color = (0, 0, 0)
        self.device = device

        self.features_dir = "object_features/resnet50/"
        self.roi_sampling_ratio = 1
        self.max_bboxes = 32

        resnet = torchvision.models.resnet50(pretrained=True)
        self.avg_pool = nn.Sequential(list(resnet.children())[-2])
        self.avg_pool.eval()

    def __len__(self):
        return len(self.mega_table)

    def __getitem__(self, idx):
        
        image_path = self.mega_table['imagePath'].iloc[idx]
        #image_name = os.path.splitext(os.path.basename(image_path))[0]

        object_id = self.mega_table['ObjectAnnotatorId'].iloc[idx]
        features_path = os.path.join(self.features_dir, object_id + ".pt")

        bb_features = torch.load(features_path).to(self.device)

        input_vector_max = np.amax(self.mega_table[self.input_vector].values, axis=0)
        input_vector = self.mega_table[self.input_vector].iloc[idx].values

        input_vector = input_vector/input_vector_max
        input_vector = torch.from_numpy(input_vector).float()

        information = self.mega_table['informationType'].iloc[idx]
        information = np.array(json.loads(information))
        information = torch.from_numpy(information)

        informativeness = self.mega_table['informativeness'].iloc[idx]
        informativeness = torch.tensor(int(informativeness))

        sharingOwner = self.mega_table['sharingOwner'].iloc[idx]
        sharingOwner = np.array(json.loads(sharingOwner))
        sharingOwner = torch.from_numpy(sharingOwner)

        sharingOthers = self.mega_table['sharingOthers'].iloc[idx]
        sharingOthers = np.array(json.loads(sharingOthers))
        sharingOthers = torch.from_numpy(sharingOthers)

        return bb_features, input_vector, information, informativeness, sharingOwner, sharingOthers

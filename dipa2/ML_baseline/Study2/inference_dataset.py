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


class ImageMaskDataset(Dataset):
    def __init__(self, mega_table, image_folder, input_vector, image_size, flip = False, save_mask = False, flip_prob = 0.5):
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

    def __len__(self):
        return len(self.mega_table)

    def __getitem__(self, idx):
        
        image_path = self.mega_table['imagePath'].iloc[idx]
        image = Image.open(os.path.join(self.image_folder, image_path)).convert('RGB')
        w, h = image.size
        ratio = min(self.image_size[0] / h, self.image_size[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = TF.resize(image, (new_h, new_w))
        image = TF.pad(image, padding=(0, 0, self.image_size[1] - new_w, self.image_size[0] - new_h), fill=self.padding_color)
        image = TF.to_tensor(image)

        trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ## generate mask
        image = trans(image)

        bboxes = json.loads(self.mega_table['bbox'].iloc[idx])
        mask = torch.zeros((self.input_dim, self.image_size[0], self.image_size[1]))
        for i, input_name in enumerate(self.input_vector):
            if self.save_mask and os.path.exists(os.path.join('./masks', input_name, self.mega_table['id'].iloc[idx] + '.pt')):
                mask[i, :, :] = torch.load(os.path.join('./masks', input_name, self.mega_table['id'].iloc[idx] + '.pt'))
            else:
                tot_num = np.amax(self.mega_table[input_name].values)
                
                for j in range(len(bboxes)):
                    bbox = bboxes[j]
                    x = bbox[0] * ratio
                    y = bbox[1] * ratio
                    w = bbox[2] * ratio
                    h = bbox[3] * ratio
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    mask[i, y:y+h, x:x+w] = self.mega_table[input_name].iloc[idx] / (tot_num + 1.0)
                if self.save_mask:
                    if not os.path.exists(os.path.join('./masks', input_name)):
                        os.mkdir(os.path.join('./masks', input_name))
                    torch.save(mask[i, :, :], os.path.join('./masks', input_name, self.mega_table['id'].iloc[idx] + '.pt'))

        if mask.nonzero().shape[0] == 0:
            print('non mask')
        if (mask > 1).any():
            print("Mask contains values greater than 1.")
        if self.flip and torch.rand(1) < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        input_vector = self.mega_table[self.input_vector].iloc[idx].values
        input_vector = torch.from_numpy(input_vector)

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

        return image, mask, information, informativeness, sharingOwner, sharingOthers

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

class VISPRDataset(Dataset):
    def __init__(self, type:str='training', image_size = (224, 224), padding_color = (0, 0 ,0)) -> None:
        super().__init__()
        self.image_folder = './vispr/train2017' if type == 'training' else './vispr/val2017'
        self.category_converter_path = './vispr/category_converter.csv'
        self.dipa_category_path = './DIPA2/category.csv'
        self.image_size = image_size
        self.padding_color = padding_color
        # read as csv, row is the original category, [1] is the category in DIPA2
        if os.path.exists(self.category_converter_path):
            with open(self.category_converter_path) as f:
                self.category_converter = pd.read_csv(f, header=None)
                # remove the first row
                self.category_converter = self.category_converter.iloc[1:]
            self.category_converter = self.category_converter.set_index(0).to_dict()[1]
        # read dipa_category as csv, row [0] is the original category, [1] is the corresponding number
        if os.path.exists(self.dipa_category_path):
            with open(self.dipa_category_path) as f:
                self.dipa_category = pd.read_csv(f, header=None)
                # remove the first row
                self.dipa_category = self.dipa_category.iloc[1:]
            self.dipa_category = self.dipa_category.set_index(0).to_dict()[1]
        print('category converter: ', self.category_converter)
        self.dataset = self.init_dataset()

    def init_dataset(self):
        files = os.listdir(self.image_folder)
        #filter file having .json extension
        files = [file for file in files if file.endswith('.json')]
        dataset = []
        for file in files:
            with open(os.path.join(self.image_folder, file)) as f:
                data = json.load(f)
                # make sure the image exists, or we continue
                if not os.path.exists(os.path.join(self.image_folder, file[:-5] + '.jpg')):
                    continue
                dataset.append({
                    'image_id': file[:-5],
                    'labels': [label for label in data['labels']],
                    'boxes': [] # this dataset does not have boxes
                })
        return dataset
    
    def count_category_for_original_data(self):
        files = os.listdir(self.image_folder)
        #filter file having .json extension
        files = [file for file in files if file.endswith('.json')]
        #set a unique set
        category = set()
        for file in files:
            with open(os.path.join(self.image_folder, file)) as f:
                data = json.load(f)
                for label in data['labels']:
                    category.add(label)
        # sort the category accord to the number, the prefix is like a01_xxxx, a02_xxxx, a03_xxxx
        category = sorted(category, key=lambda x: x.split('_')[0][1:])
        print('len of category: ', len(category))
        print('category: ', category)

    def __getitem__(self, index):
        # read image, and convert to tensor
        # read label, covert to DIPA2 label
        # make sure the channel is 3 when reading image
        image = Image.open(os.path.join(self.image_folder, self.dataset[index]['image_id'] + '.jpg')).convert('RGB')
        w, h = image.size
        ratio = min(self.image_size[0] / h, self.image_size[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = TF.resize(image, (new_h, new_w))
        image = TF.pad(image, padding=(0, 0, self.image_size[1] - new_w, self.image_size[0] - new_h), fill=self.padding_color)
        image = TF.to_tensor(image)
        # normalize image
        trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = trans(image)

        # convert category to DIPA2 category
        labels = np.zeros(23, dtype=np.uint8)
        # convert original category to dipaCategory, set the label to 1 if the category is in the image, according to numbers of self.dipa_category
        for label in self.dataset[index]['labels']:
            if label in self.category_converter:
                labels[int(self.dipa_category[self.category_converter[label]])] = 1
            else:
                Exception('label not found in category converter: ', label)
        labels = torch.from_numpy(labels)
        return image, labels
    
    def __len__(self):
        return len(self.dataset)
    
if __name__ == '__main__':
    dataset = VISPRDataset()
    dataset.count_category_for_original_data()
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
import cv2

class VIZWIZDataset(Dataset):
    def __init__(self, type:str='training', image_size = (224, 224), padding_color = (0, 0 ,0)) -> None:
        super().__init__()
        self.image_folder = './vizwiz/Filling_Images'
        self.annotation_path = './vizwiz/Annotations/train.json' if type == 'training' else './vizwiz/Annotations/val.json'
        self.category_converter_path = './vizwiz/category_converter.csv'
        self.dipa_category_path = './DIPA2/category.csv'
        self.image_size = image_size
        self.padding_color = padding_color
        # read as csv, row [0] is the original category, [1] is the category in DIPA2
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
        with open(self.annotation_path) as f:
            data = json.load(f)
            dataset = []
            for image in data:
                if not image['private']:
                    continue
                # make sure the image exists, or we continue
                # the annotation file record photos as jpg, but the actual file is png :) in VizWiz
                if not os.path.exists(os.path.join(self.image_folder, image['image'][:-4] + '.png')):
                    print('image not exists: ', os.path.join(self.image_folder, image['image']))
                    continue
                dataset.append({
                    'image_id': image['image'][:-4] + '.png',
                    'labels': [label['class'] for label in image['private_regions']],
                    'boxes': [label['polygon'] for label in image['private_regions']] # we do not use boxes in the original inference, the model will only predict the privacy category in image to align with VISPR.
                })
        return dataset

    def count_category_for_original_data(self):
        with open(self.annotation_path) as f:
            data = json.load(f)
            category = set()
            for image in data:
                if not image['private']:
                    continue
                for label in image['private_regions']:
                    category.add(label['class'])
            print('len of category: ', len(category))
            print('category: ', category)
    def count_bounding_box_distribution(self):
        # we also fetech the data from vizwiz-priv dataset
        relative_sizes = []
        relative_position = []
        width_height_ratio = []
        file_path = os.path.join('vizwiz', 'Annotations', 'dataset.json')
        with open(file_path) as f:
            data = json.load(f)
            for item in data:
                if item['private']:
                    image_path = os.path.join('vizwiz', 'Filling_Images', item['image'][:-4] + '.png')
                    image = Image.open(image_path)
                    image_width, image_height = image.size
                    for regoin in item['private_regions']:
                        # make the polygon into rectangle by cv2.boundingRect
                        x, y, w, h = cv2.boundingRect(np.array(regoin['polygon']))
                        size = w * h
                        
                        center_x = image_width / 2
                        center_y = image_height / 2
                        relative_x = (x + w / 2 - center_x) / image_width
                        relative_y = (y + h / 2 - center_y) / image_height
                        if relative_x > 0.5 or relative_x < -0.5 or relative_y > 0.5 or relative_y < -0.5:
                            print('relative_x or relative_y out of range')
                            continue
                        relative_position.append([relative_x, relative_y])
                        width_height_ratio.append(w / h)
                        relative_sizes.append(size / (image_width * image_height))
        # save the data
        relative_sizes = np.array(relative_sizes)
        relative_position = np.array(relative_position)
        width_height_ratio = np.array(width_height_ratio)
        np.save('vizwiz_relative_sizes.npy', relative_sizes)
        np.save('vizwiz_relative_position.npy', relative_position)
        np.save('vizwiz_width_height_ratio.npy', width_height_ratio)

    def __getitem__(self, index):
        # read image, and convert to tensor
        # read label, covert to DIPA2 label

        image = Image.open(os.path.join(self.image_folder, self.dataset[index]['image_id'])).convert('RGB')
        # resize image
        w, h = image.size
        ratio = min(self.image_size[0] / h, self.image_size[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = TF.resize(image, (new_h, new_w))
        image = TF.pad(image, padding=(0, 0, self.image_size[1] - new_w, self.image_size[0] - new_h), fill=self.padding_color)
        image = TF.to_tensor(image)
        # normalize image
        trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = trans(image)
        # in DIPA2, we have 23 predefined categories with 'others'. So, the label will be a 22-dim vector.
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
    dataset = VIZWIZDataset()
    dataset.count_category_for_original_data()
    dataset.count_bounding_box_distribution()
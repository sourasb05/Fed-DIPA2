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

class DIPADataset(Dataset):
    def __init__(self, type:str='training', image_size = (224, 224), padding_color = (0, 0 ,0)) -> None:
        super().__init__()
        self.annotation_path = '../../annotations.csv'
        self.image_root_path = '../../images'
        self.dipa_category_path = './DIPA2/category.csv'
        self.type = type
        self.image_size = image_size
        self.padding_color = padding_color
        # read dipa_category as csv, row [0] is the original category, [1] is the corresponding number
        if os.path.exists(self.dipa_category_path):
            with open(self.dipa_category_path) as f:
                self.dipa_category = pd.read_csv(f, header=None)
                # remove the first row
                self.dipa_category = self.dipa_category.iloc[1:]
            self.dipa_category = self.dipa_category.set_index(0).to_dict()[1]
        self.dataset = self.init_dataset()
        
    def init_dataset(self):
        #read the annotations.csv to get dataset
        #divide the dataset into training, validation, testing 65-10-25
        #return the dataset
        with open(self.annotation_path) as f:
            mega_table = pd.read_csv(f)
            dataset = []
            # group all the rows by image_id, aggregate the labels and boxes
            for imagePath, group in mega_table.groupby('imagePath'):
                dataset.append({
                    'image_id': imagePath,
                    'labels': group['DIPACategory'].tolist(),
                    'boxes': []
                })
        #divide the dataset into training, validation, testing 65-10-25, seed = 42
        np.random.seed(42)
        np.random.shuffle(dataset)
        if self.type == 'training':
            dataset = dataset[:int(len(dataset)*0.65)]
        elif self.type == 'validation':
            dataset = dataset[int(len(dataset)*0.65):int(len(dataset)*0.75)]
        elif self.type == 'testing':
            dataset = dataset[int(len(dataset)*0.75):]
        else:
            raise Exception('type should be training, validation or testing')
        
        # calculate the ratio of positive and negative samples in labels
        positive = 0
        negative = 0
        for data in dataset:
            negative += 23
            for label in data['labels']:
                # if the label is in dipa_category, then it is a positive sample, then negative - 1, and positive + 1
                if label in self.dipa_category:
                    negative -= 1
                    positive += 1

        # save the number in a json file
        with open('./positive_negative_{}.json'.format(self.type), 'w') as f:
            json.dump({'positive': positive, 'negative': negative, 'ratio': positive/(positive + negative)}, f)

        return dataset
    
    def count_category_for_original_data(self):
        mega_table = pd.read_csv(self.annotation_path)
        category = set()
        for index, row in mega_table.iterrows():
            category.add(row['DIPACategory'])
        print('len of category: ', len(category))
        print('category: ', category)
    

    def __getitem__(self, index):
        # read image, and convert to tensor
        # read label, covert to DIPA2 label

        image = Image.open(os.path.join(self.image_root_path, self.dataset[index]['image_id'])).convert('RGB')
        w, h = image.size
        ratio = min(self.image_size[0] / h, self.image_size[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = TF.resize(image, (new_h, new_w))
        image = TF.pad(image, padding=(0, 0, self.image_size[1] - new_w, self.image_size[0] - new_h), fill=self.padding_color)
        image = TF.to_tensor(image)
        # normalize image
        trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = trans(image)

        labels = np.zeros(23, dtype=np.uint8)
        
        # convert original category to dipaCategory, set the label to 1 if the category is in the image, according to numbers of self.dipa_category
        for label in self.dataset[index]['labels']:
            if label in self.dipa_category:
                labels[int(self.dipa_category[label])] = 1
            else:
                Exception('label not found in category converter: ', label)
        labels = torch.from_numpy(labels)
        return image, labels

    def __len__(self):
        return len(self.dataset)
if __name__ == '__main__':
    dataset = DIPADataset()
    dataset.count_category_for_original_data()
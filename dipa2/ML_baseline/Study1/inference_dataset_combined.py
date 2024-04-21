import torch
from torch.utils.data import Dataset
from inference_dataset_VizWiz import VIZWIZDataset
from inference_dataset_VISPR import VISPRDataset


class CombinedDataset(Dataset):
    def __init__(self, type='training', image_size=(224, 224), padding_color=(0, 0, 0)):
        self.mode = type
        self.vispr_dataset = VISPRDataset(type=type, image_size=image_size, padding_color=padding_color)
        self.vizwiz_dataset = VIZWIZDataset(type=type, image_size=image_size, padding_color=padding_color)

    def __getitem__(self, index):
        if index < len(self.vispr_dataset):
            return self.vispr_dataset[index]
        else:
            return self.vizwiz_dataset[index - len(self.vispr_dataset)]

    def __len__(self):
        return len(self.vispr_dataset) + len(self.vizwiz_dataset)

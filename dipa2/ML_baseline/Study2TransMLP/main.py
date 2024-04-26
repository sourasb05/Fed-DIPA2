import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from inference_dataset import ImageMaskDataset
from inference_model import BaseModel

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import json
import sys

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime 

def GetDateAndTime():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H:%M:%S')

def l1_distance_loss(prediction, target):
    loss = np.abs(prediction - target)
    return np.mean(loss)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bigfives = ["extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness"]
    basic_info = [ "age", "gender", 'nationality', 'frequency']
    category = ['category']
    privacy_metrics = ['informationType', 'informativeness', 'sharingOwner', 'sharingOthers']

    mega_table = pd.read_csv('./annotations_filtered_bbox.csv')

    description = {'informationType': ['It tells personal information', 'It tells location of shooting',
        'It tells individual preferences/pastimes', 'It tells social circle', 
        'It tells others\' private/confidential information', 'Other things'],
        'informativeness':['Strongly disagree','Disagree','Slightly disagree','Neither',
        'Slightly agree','Agree','Strongly agree'],
        'sharingOwner': ['I won\'t share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'],
        'sharingOthers': ['I won\'t allow others to share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients']}
    
    encoder = LabelEncoder()
    mega_table['category'] = encoder.fit_transform(mega_table['category'])
    mega_table['gender'] = encoder.fit_transform(mega_table['gender'])
    mega_table['platform'] = encoder.fit_transform(mega_table['platform'])
    mega_table['originalDataset'] = encoder.fit_transform(mega_table['originalDataset'])
    mega_table['nationality'] = encoder.fit_transform(mega_table['nationality'])

    input_channel = []
    input_channel.extend(basic_info)
    input_channel.extend(category)
    input_channel.extend(bigfives)
    input_dim = len(input_channel)
    output_name = privacy_metrics
    output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}

    model = BaseModel(input_dim= input_dim)

    image_size = (224, 224)
    image_folder = './images/'

    num_rows = len(mega_table)

    test_set = True

    if not test_set:
        train_size = int(0.8 * num_rows)
        test_size = num_rows - train_size
        # Split the dataframe into two
        train_df = mega_table.sample(n=train_size, random_state=0)
        val_df = mega_table.drop(train_df.index)
    else:
        # Dataset Split Percentage
        train_per, val_per, test_per = 65, 10, 25
        train_size = int((train_per/100.0) * num_rows)
        val_size = int((val_per/100.0) * num_rows)
        test_size = num_rows - train_size - val_size

        train_df = mega_table.sample(n=train_size, random_state=0)
        rem_df = mega_table.drop(train_df.index)
        val_df = rem_df.sample(n=val_size, random_state=0)
        test_df = rem_df.drop(val_df.index)

    train_dataset = ImageMaskDataset(train_df, image_folder, input_channel, image_size, flip = True, device=device)
    val_dataset = ImageMaskDataset(val_df, image_folder, input_channel, image_size, device=device)

    train_loader = DataLoader(train_dataset, batch_size=256,
                              generator=torch.Generator(device='cuda'),
                              shuffle=True)
    val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'), batch_size=32)

    if test_set:
        test_dataset = ImageMaskDataset(test_df, image_folder, input_channel, image_size)
        test_loader = DataLoader(test_dataset, generator=torch.Generator(device='cuda'), batch_size=32)
        

    wandb_logger = WandbLogger(project="DIPA2-MLP", name = 'R50_' + GetDateAndTime())#, mode="disabled")

    checkpoint_callback = ModelCheckpoint(dirpath='./ML baseline/Study2-TransMLP/models/Resnet50/', save_last=True, monitor='val loss')
    trainer = pl.Trainer(accelerator='gpu', devices=[0],logger=wandb_logger, 
    auto_lr_find=True, max_epochs = 100, callbacks=[checkpoint_callback])
    lr_finder = trainer.tuner.lr_find(model, train_loader)
    model.hparams.learning_rate = lr_finder.suggestion()
    print(f'lr auto: {lr_finder.suggestion()}')
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader) 
    
    threshold = 0.5
    average_method = 'weighted'
    acc = [Accuracy(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    pre = [Precision(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    rec = [Recall(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    f1 = [F1Score(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    conf = [ConfusionMatrix(task="multilabel", num_labels=output_dim) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    distance = 0.0
    
    model.to('cuda')
    for i, vdata in enumerate(val_loader):
        image, mask, information, informativeness, sharingOwner, sharingOthers = vdata
        y_preds = model(image.to('cuda'), mask.to('cuda'))
        #print(y_preds[:, :6].shape, information.shape)

        acc[0].update(y_preds[:, :6], information.to('cuda'))
        pre[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        rec[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        f1[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        conf[0].update(y_preds[:, :6], information.to('cuda'))

        distance += l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())

        acc[1].update(y_preds[:, 7:14], sharingOwner.to('cuda'))
        pre[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        rec[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        f1[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        conf[1].update(y_preds[:, 7:14], sharingOwner.to('cuda'))

        acc[2].update(y_preds[:, 14:21], sharingOthers.to('cuda'))
        pre[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        rec[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        f1[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        conf[2].update(y_preds[:, 14:21], sharingOthers.to('cuda'))

    distance = distance / len(val_loader)

    pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in acc], 
                   'Precision' : [i.compute().detach().cpu().numpy() for i in pre], 
                   'Recall': [i.compute().detach().cpu().numpy() for i in rec], 
                   'f1': [i.compute().detach().cpu().numpy() for i in f1]}

    print(pandas_data)
    for i, k in enumerate(output_channel.keys()):
        print("%.02f %.02f %.02f" % (pandas_data["Accuracy"][i], pandas_data["Precision"][i], pandas_data["Recall"][i]), end=" ")

    # df = pd.DataFrame(pandas_data, index=output_channel.keys())
    # print(df.round(3))
    # df.to_csv('./result.csv', index =False)
    # with open('./distance', 'w') as w:
    #     w.write(str(distance))
    
    print('informativenss distance: ', distance)
    if 'informativeness' in output_channel.keys():
        print('informativenss distance: ', distance)

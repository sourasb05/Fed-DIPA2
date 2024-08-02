import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from inference_dataset_DIPA2 import DIPADataset
from inference_dataset_VizWiz import VIZWIZDataset
from inference_dataset_VISPR import VISPRDataset
from inference_dataset_combined import CombinedDataset
from inference_model import BaseModel

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import json

from pytorch_lightning.callbacks import ModelCheckpoint


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, default='None', help='vizwiz, vispr')
max_epochs = 100
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def pretrain_with_vispr(model):
    checkpoint_callback = ModelCheckpoint(
        dirpath='./models/Resnet50/', 
        filename='vispr_pretrain-{epoch:02d}-{val_loss:.2f}', 
        save_top_k=1, 
        monitor='val_loss', 
        mode='min',
        save_last=False
    )
    vispr_dataset = VISPRDataset()
    val_vispr_dataset = VISPRDataset(type='validation')
    train_loader = DataLoader(vispr_dataset, batch_size=64, generator=torch.Generator(device='cuda'), shuffle=True)
    val_loader = DataLoader(val_vispr_dataset, generator=torch.Generator(device='cuda'), batch_size=32)
    # we do not set val and test loader here
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=[0],
        auto_lr_find=True, 
        max_epochs = max_epochs, 
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    best_model = BaseModel.load_from_checkpoint(best_model_path)

    return best_model



def pretrain_with_vizwiz(model):
    checkpoint_callback = ModelCheckpoint(
        dirpath='./models/Resnet50/', 
        filename='vizwiz_pretrain-{epoch:02d}-{val_loss:.2f}', 
        save_top_k=1, 
        monitor='val_loss', 
        mode='min',
        save_last=False
    )
    vizwiz_dataset = VIZWIZDataset()
    val_vizwiz_dataset = VIZWIZDataset(type='validation')
    train_loader = DataLoader(vizwiz_dataset, batch_size=64, generator=torch.Generator(device='cuda'), shuffle=True)
    val_loader = DataLoader(val_vizwiz_dataset, generator=torch.Generator(device='cuda'), batch_size=32)
    # we do not set val and test loader here
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=[0],
        auto_lr_find=True, 
        max_epochs = max_epochs, 
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    best_model = BaseModel.load_from_checkpoint(best_model_path)

    return best_model

def pretrain_on_both(model):
    # pretrain on vizwiz and vispr
    checkpoint_callback = ModelCheckpoint(
        dirpath='./models/Resnet50/', 
        filename='both_pretrain-{epoch:02d}-{val_loss:.2f}', 
        save_top_k=1, 
        monitor='val_loss', 
        mode='min',
        save_last=False
    )
    train_CombinedDataset = CombinedDataset()
    val_CombinedDataset = CombinedDataset(type='validation')
    train_loader = DataLoader(train_CombinedDataset, batch_size=64, generator=torch.Generator(device='cuda'), shuffle=True)
    val_loader = DataLoader(val_CombinedDataset, generator=torch.Generator(device='cuda'), batch_size=32)
    # we do not set val and test loader here
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=[0],
        auto_lr_find=True, 
        max_epochs = max_epochs, 
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    best_model = BaseModel.load_from_checkpoint(best_model_path)

    return best_model


if __name__ == '__main__':
    # Prepare arguments
    args = parser.parse_args()
    results = {}
    models = []
    if args.pretrain == 'vizwiz':
        model = BaseModel()
        model = pretrain_with_vizwiz(model)
        models.append(('vizwiz_pretrain', model))

    elif args.pretrain == 'vispr':
        model = BaseModel()
        model = pretrain_with_vispr(model)
        models.append(('vispr_pretrain', model))

    elif args.pretrain == 'all':
        model_vizwiz = BaseModel()
        model_vizwiz = pretrain_with_vizwiz(model_vizwiz)
        models.append(('vizwiz_pretrain', model_vizwiz))

        model_vispr = BaseModel()
        model_vispr = pretrain_with_vispr(model_vispr)
        models.append(('vispr_pretrain', model_vispr))

        model_no_pretrain = BaseModel()
        models.append(('no_pretrain', model_no_pretrain))

        model_both = BaseModel()
        model_both = pretrain_on_both(model_both)
        models.append(('both_pretrain', model_both))

    elif args.pretrain == 'None':
        model = BaseModel()
        models.append(('no_pretrain', model))


    # Prepare DIPA dataset
    image_size = (224, 224)
    train_dataset = DIPADataset(image_size=image_size, type='training')
    val_dataset =  DIPADataset(image_size=image_size, type='validation')
    test_dataset = DIPADataset(image_size=image_size, type='testing')

    train_loader = DataLoader(train_dataset, batch_size=64, generator=torch.Generator(device='cuda'), shuffle=True)
    val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'), batch_size=32)
    test_loader =  DataLoader(test_dataset, generator=torch.Generator(device='cuda'), batch_size=32)

    # Prepare figure for precision-recall curve
    plt.figure()

    # For each model, train it with DIPA dataset and evaluate it
    for model_name, model in models:
        
        checkpoint_callback = ModelCheckpoint(
                dirpath='./models/Resnet50/', 
                filename=f'{model_name}-{{epoch:02d}}-{{val_loss:.2f}}', 
                save_top_k=1, 
                monitor='val_loss', 
                mode='min', 
                save_last=False,
        )

        trainer = pl.Trainer(
                            accelerator='gpu', 
                            devices=[0],
                            auto_lr_find=True, 
                            max_epochs = max_epochs, 
                            callbacks=[checkpoint_callback]
        )
        lr_finder = trainer.tuner.lr_find(model, train_loader)
        model.hparams.learning_rate = lr_finder.suggestion()
        print(f'{model_name} lr auto: {lr_finder.suggestion()}')

        if model_name in ['vizwiz_pretrain', 'vispr_pretrain', 'both_pretrain']:
            # we also record the trainer test without training on DIPA2
            trainer.test(model, test_loader)  # assuming the model has a test_step implemented
            # save the precision recall curve
            y_test = []
            y_score = []
            for images, labels in test_loader:
                with torch.no_grad():
                    outputs = model(images)
                y_test.append(labels.cpu().numpy())
                y_score.append(outputs.cpu().numpy())
            #save in plt
            y_test = np.concatenate(y_test, axis=0)
            y_score = np.concatenate(y_score, axis=0)
            precision, recall, _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
            average_precision = average_precision_score(y_test, y_score, average="micro")
            # save the results in dict
            results[model_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'average_precision': average_precision.item()  # if average_precision is also a numpy array
            }
            plt.plot(recall, precision, alpha=1.0,
                    label='{} (AP = {:0.2f})'
                        ''.format(model_name + '(without training on DIPA2)', average_precision))
            
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader) 

        # Compute precision and recall
        y_test = []
        y_score = []

        best_model_path = checkpoint_callback.best_model_path
        best_model = BaseModel.load_from_checkpoint(best_model_path)

        # now use the best_model for testing
        trainer.test(best_model, test_loader)

        for images, labels in test_loader:
            with torch.no_grad():
                outputs = model(images)
            y_test.append(labels.cpu().numpy())
            y_score.append(outputs.cpu().numpy())

        y_test = np.concatenate(y_test, axis=0)
        y_score = np.concatenate(y_score, axis=0)

        precision, recall, _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        average_precision = average_precision_score(y_test, y_score, average="micro")
        results[model_name] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'average_precision': average_precision.item()  # if average_precision is also a numpy array
        }

        # Plot Precision-Recall curve
        plt.plot(recall, precision, alpha=1.0,
                 label='{} (AP = {:0.2f})'
                       ''.format(model_name, average_precision))

    # Set labels and title for the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curves (micro-averaged)')
    plt.legend(loc="lower right")
    # plt.show()
    # Save the plot
    plt.savefig('precision_recall_curve.png')

    # Save the results in json
    with open('precision_recall_curve.json', 'w') as f:
        json.dump(results, f)




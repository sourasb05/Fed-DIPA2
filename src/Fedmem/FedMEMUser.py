import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,  TensorDataset
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import glob
import copy
import pandas as pd
import numpy as np
from src.Optimizer.Optimizer import Fedmem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from dipa2.ML_baseline.Study2.inference_dataset import ImageMaskDataset
# from dipa2.ML_baseline.Study2.inference_model import BaseModel
from src.TrainModels.trainmodels import BaseModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import io
from PIL import Image
import json
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Fedmem_user():

    def __init__(self,device, args, id, exp_no, current_directory):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device
        
        self.id = id  # integer
        self.batch_size = args.batch_size
        self.exp_no = exp_no
        self.current_directory = current_directory
        """
        Hyperparameters
        """
        self.learning_rate = args.alpha
        self.local_iters = args.local_iters
        self.eta = args.eta
        self.global_model_name = args.model_name
        self.algorithm = args.algorithm
        self.cluster_type = args.cluster
        self.data_silo = args.data_silo
        self.target = args.target
        self.num_users = args.total_users * args.users_frac 
        self.fixed_user_id = args.fixed_user_id
        self.country = "japan"
        
        self.distance = 0.0
        
        
        self.bigfives = ["extraversion", "agreeableness", "conscientiousness",
                         "neuroticism", "openness"]
        self.basic_info = [ "age", "gender", 'nationality', 'frequency']
        self.category = ['category']
        self.privacy_metrics = ['informationType', 'informativeness', 'sharingOwner', 'sharingOthers']
        self.mega_table = pd.read_csv(current_directory + '/clients/new_annotations_annotator' + str(self.id) + '.csv')
        
        self.description = {'informationType': ['It tells personal information', 
                                                'It tells location of shooting',
                                                'It tells individual preferences/pastimes',
                                                'It tells social circle', 
                                                'It tells others\' private/confidential information',
                                                'Other things'],
                                                'informativeness':['Strongly disagree',
                                                                   'Disagree',
                                                                   'Slightly disagree',
                                                                   'Neither', 'Slightly agree',
                                                                   'Agree','Strongly agree'],
                            'sharingOwner': ['I won\'t share it', 
                                             'Close relationship',
                                             'Regular relationship',
                                            'Acquaintances', 
                                            'Public', 
                                            'Broadcast program', 
                                            'Other recipients'],
                            'sharingOthers': ['I won\'t allow others to share it',
                                            'Close relationship',
                                            'Regular relationship',
                                            'Acquaintances',
                                            'Public',
                                            'Broadcast program',
                                            'Other recipients']}
    
        self.encoder = LabelEncoder()
        self.mega_table['category'] = self.encoder.fit_transform(self.mega_table['category'])
        self.mega_table['gender'] = self.encoder.fit_transform(self.mega_table['gender'])
        self.mega_table['platform'] = self.encoder.fit_transform(self.mega_table['platform'])
        self.mega_table['originalDataset'] = self.encoder.fit_transform(self.mega_table['originalDataset'])
        self.mega_table['nationality'] = self.encoder.fit_transform(self.mega_table['nationality'])

        self.input_channel = []
        self.input_channel.extend(self.basic_info)
        self.input_channel.extend(self.category)
        self.input_channel.extend(self.bigfives)
        self.input_dim = len(self.input_channel)
        self.output_name = self.privacy_metrics
        self.output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}


        self.local_model = BaseModel(input_dim=self.input_dim).to(self.device)
        self.eval_model = copy.deepcopy(self.local_model)

        image_size = (224, 224)
        image_folder = self.current_directory + '/dipa2/images/'

        num_rows = len(self.mega_table)
        train_size = int(0.8 * num_rows)
        test_size = num_rows - train_size
        self.train_samples = train_size
        # Split the dataframe into two
        train_df = self.mega_table.sample(n=train_size, random_state=0)
        val_df = self.mega_table.drop(train_df.index)

        train_dataset = ImageMaskDataset(train_df, image_folder, self.input_channel, image_size, flip = True)
        val_dataset = ImageMaskDataset(val_df, image_folder, self.input_channel, image_size)    

        self.train_loader = DataLoader(train_dataset, batch_size=96, generator=torch.Generator(device='cuda'), shuffle=True)
        self.val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'), batch_size=32)
    
        self.wandb_logger = WandbLogger(project="DIPA2.0 baseline", name = 'Resnet50')
        self.checkpoint_callback = ModelCheckpoint(dirpath='./ML baseline/Study2/models/Resnet50/', save_last=True, monitor='val loss')

      
        
        self.optimizer= torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate)
        
        # metrics

        threshold = 0.5
        average_method = 'micro'
        self.acc = [Accuracy(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        self.pre = [Precision(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        self.rec = [Recall(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        self.f1 = [F1Score(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        self.conf = [ConfusionMatrix(task="multilabel", num_labels=output_dim) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        
        self.global_acc = [Accuracy(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        self.global_pre = [Precision(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        self.global_rec = [Recall(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        self.global_f1 = [F1Score(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        self.global_conf = [ConfusionMatrix(task="multilabel", num_labels=output_dim) \
                for i, (output_name, output_dim) in enumerate(self.output_channel.items())]
        
        # print(self.global_acc)
        
        # input("press")
        
        

        
    def set_parameters(self, cluster_model):
        for param, glob_param in zip(self.local_model.parameters(), cluster_model):
            param.data = glob_param.data.clone()
            # print(f"user {self.id} parameters : {param.data}")
        # input("press")
            
    def get_parameters(self):
        return self.local_model.parameters()

    def update_parameters(self, new_params):
        for param, new_param in zip(self.eval_model.parameters(), new_params):
            param.data = new_param.data.clone()


    def test(self, global_model, t):
        # Set the model to evaluation mode
        self.local_model.eval()
        self.update_parameters(global_model)
        total_loss=0.0
        distance = 0.0
        for i, vdata in enumerate(self.val_loader):
            image, mask, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(image.to(self.device), mask.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            total_loss += loss.item()
            
            print(y_preds[:, :6].shape, information.shape)

            self.global_acc[0].update(y_preds[:, :6], information.to(self.device))
            self.global_pre[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.global_rec[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.global_f1[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.global_conf[0].update(y_preds[:, :6], information.to(self.device))
            
            distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())
            # print(f"disance: {self.distance}")

            self.global_acc[1].update(y_preds[:, 7:14], sharingOwner.to(self.device))
            self.global_pre[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to(self.device))
            self.global_rec[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to(self.device))
            self.global_f1[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to(self.device))
            self.global_conf[1].update(y_preds[:, 7:14], sharingOwner.to(self.device))

            self.global_acc[2].update(y_preds[:, 14:21], sharingOthers.to(self.device))
            self.global_pre[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to(self.device))
            self.global_rec[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to(self.device))
            self.global_f1[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to(self.device))
            self.global_conf[2].update(y_preds[:, 14:21], sharingOthers.to(self.device))

            
        distance = distance / len(self.val_loader)

        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.global_acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.global_pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.global_rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.global_f1]}
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
        #print(pandas_data)
        

        avg_loss = total_loss / len(self.val_loader)
        # print(f"Global iter {t}: Validation loss: {avg_loss}")
        # print(f"distance: {distance}")
        # input("press")
        return avg_loss, distance, pandas_data
        
    def l1_distance_loss(self, prediction, target):
        loss = np.abs(prediction - target)
        return np.mean(loss)
        
    def evaluate_model(self):
        self.local_model
        total_loss=0.0
        for i, vdata in enumerate(self.val_loader):
            image, mask, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(image.to(self.device), mask.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            total_loss += loss.item()
            
            print(y_preds[:, :6].shape, information.shape)

            self.acc[0].update(y_preds[:, :6], information.to(self.device))
            self.pre[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.rec[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.f1[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.conf[0].update(y_preds[:, :6], information.to(self.device))
            
            self.distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())
            # print(f"disance: {self.distance}")

            self.acc[1].update(y_preds[:, 7:14], sharingOwner.to(self.device))
            self.pre[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to(self.device))
            self.rec[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to(self.device))
            self.f1[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to(self.device))
            self.conf[1].update(y_preds[:, 7:14], sharingOwner.to(self.device))

            self.acc[2].update(y_preds[:, 14:21], sharingOthers.to(self.device))
            self.pre[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to(self.device))
            self.rec[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to(self.device))
            self.f1[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to(self.device))
            self.conf[2].update(y_preds[:, 14:21], sharingOthers.to(self.device))

            
        self.distance = self.distance / len(self.val_loader)

        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.f1]}
       
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}

        print(pandas_data)

        avg_loss = total_loss / len(self.val_loader)
        # print(f"Validation loss: {avg_loss}")
        # print(f"distance: {self.distance}")
        # input("press")
        # print(f"Epoch : {iter}  loss: {loss.item()} acc: {self.acc} pre: {self.pre} f1: {self.f1} conf: {self.conf}")


        # df = pd.DataFrame(pandas_data, index=output_channel.keys())
        # print(df.round(3))
        # df.to_csv('./result.csv', index =False)
        # with open('./distance', 'w') as w:
        #     w.write(str(distance))
        
        # if 'informativeness' in self.output_channel.keys():
        #    print('informativenss distance: ', distance)
        
    def train(self):
        print(f"user id : {self.id}")
        
        wandb_logger = WandbLogger(project="DIPA2.0 baseline", name = 'Resnet50')
        checkpoint_callback = ModelCheckpoint(dirpath=self.current_directory + '/models/local_models/annotator_' + str(),save_last=True, monitor='val loss')
        self.local_model.train()
        for iter in range(self.local_iters):
            for batch in self.train_loader:
                image, mask, information, informativeness, sharingOwner, sharingOthers = batch
                self.optimizer.zero_grad()
                y_preds = self.local_model(image.to(self.device), mask.to(self.device))
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                loss.backward()
                self.optimizer.step()
                # print(f"Epoch : {iter} Training loss: {loss.item()}")
                # self.distance = 0.0
                # self.evaluate_model()

        
        

        # self.distance = 0.0
    
                 

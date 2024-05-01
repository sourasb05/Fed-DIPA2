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
from dipa2.ML_baseline.Study2TransMLP.inference_dataset import ImageMaskDataset
from src.TrainModels.trainmodels import PrivacyModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import io
from PIL import Image
import json
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError
import wandb
from sklearn.metrics import mean_absolute_error
import sys
import shutil
import math

class UserAvg():

    def __init__(self,device, args, id, exp_no, current_directory, wandb):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device
        self.wandb = wandb
        
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
        self.algorithm = args.algorithm
        self.fixed_user_id = args.fixed_user_id
        self.country = args.country
        
        self.distance = 0.0
        
        
        self.bigfives = ["extraversion", "agreeableness", "conscientiousness",
                         "neuroticism", "openness"]
        self.basic_info = [ "age", "gender", 'nationality', 'frequency']
        self.category = ['category']
        self.privacy_metrics = ['informationType', 'informativeness', 'sharingOwner', 'sharingOthers']
        self.mega_table = pd.read_csv(current_directory + '/feature_clients/annotations_annotator' + str(self.id) + '.csv')
        
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

        self.local_model = PrivacyModel(input_dim=self.input_dim).to(self.device)
        self.eval_model = copy.deepcopy(self.local_model)

        image_size = (224, 224)
        feature_folder = self.current_directory + '/object_features/resnet50/'

        # Dataset Allocation
        num_rows = len(self.mega_table)

        if num_rows <= 3:
            self.valid = False
            return None
        else:
            self.valid = True

        train_per, val_per, test_per = 65, 10, 25
        train_size = math.floor((train_per/100.0) * num_rows)
        val_size = math.ceil((val_per/100.0) * num_rows)
        test_size = num_rows - train_size - val_size

        train_df = self.mega_table.sample(n=train_size, random_state=0)
        rem_df = self.mega_table.drop(train_df.index)
        val_df = rem_df.sample(n=val_size, random_state=0)
        test_df = rem_df.drop(val_df.index)

        dataset_files_dir = "dataset_files/%s/" % self.algorithm
        os.makedirs(dataset_files_dir, exist_ok=True)

        train_df.to_csv("%s/train_%d.csv" % (dataset_files_dir, self.id), index=False)
        val_df.to_csv("%s/val_%d.csv" % (dataset_files_dir, self.id), index=False)

        train_dataset = ImageMaskDataset(train_df, feature_folder, self.input_channel, image_size, flip = True)
        val_dataset = ImageMaskDataset(val_df, feature_folder, self.input_channel, image_size)
        test_dataset = ImageMaskDataset(test_df, feature_folder, self.input_channel, image_size)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, generator=torch.Generator(device='cuda'), shuffle=True)
        self.trainloaderfull = DataLoader(train_dataset, batch_size=len(train_dataset), generator=torch.Generator(device='cuda'), shuffle=True)
        self.val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'), batch_size=len(val_dataset))
        self.test_loader = DataLoader(test_dataset, generator=torch.Generator(device='cuda'), batch_size=len(test_dataset))
        # Dataset Allocation ends

        self.train_samples = train_size
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
        
        if args.test:
            self.load_model()

    def load_model(self):
        models_dir = "./models/FedAvg/global_model/"
        model_state_dict = torch.load(os.path.join(models_dir, "server_checkpoint.pt"))["model_state_dict"]
        self.local_model.load_state_dict(model_state_dict)
        self.local_model.eval()
        
    def set_parameters(self, cluster_model):
        for param, glob_param in zip(self.local_model.parameters(), cluster_model.parameters()):
            param.data = glob_param.data.clone()
            # print(f"user {self.id} parameters : {param.data}")
        # input("press")
            
    def get_parameters(self):
        return self.local_model.parameters()

    def update_parameters(self, new_params):
        for param, new_param in zip(self.local_model.parameters(), new_params):
            param.data = new_param.data.clone()

    def train_evaluation(self, global_model, t):
        # Set the model to evaluation mode
        self.local_model.eval()
        self.update_parameters(global_model)
        total_loss=0.0
        distance = 0.0
        for i, vdata in enumerate(self.trainloaderfull):
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            total_loss += loss.item()/len(features)
            
            # print(y_preds[:, :6].shape, information.shape)

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

            # MAE calculation
                
            true_values = informativeness.cpu().detach().numpy()
            # print(f"true values : {true_values}")
            predicted_values = y_preds[:, 6].cpu().detach().numpy()
            # print(f"predicted values : {predicted_values}")
            mae = mean_absolute_error(true_values, predicted_values)
            
        # print(f"MAE : {mae}")
            
        mae = mae/len(self.val_loader)
            
        distance = distance / len(self.val_loader)

        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.global_acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.global_pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.global_rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.global_f1]}
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
        #print(pandas_data)
        
        # print(f"Global iter {t}: Validation loss: {avg_loss}")
        # print(f"distance: {distance}")
        
        return total_loss, distance, pandas_data, mae
    
    def test_eval(self):
        self.local_model.eval()

        results = []
        for i, vdata in enumerate(self.val_loader):
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
        return results

    def test(self, global_model=None, t=-1):
        # Set the model to evaluation mode
        self.local_model.eval()

        if global_model != None:
            self.update_parameters(global_model)

        total_loss=0.0
        distance = 0.0
        mae = 0.0
        for i, vdata in enumerate(self.val_loader):
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            
            # print("features", features)
            # print("additional_information", additional_information)
            y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            total_loss += loss.item()/len(features)
            
            # (y_preds[:, :6].shape, information.shape)

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

            # MAE calculation
                
            true_values = informativeness.cpu().detach().numpy()
            # print(f"true values : {true_values}")
            predicted_values = y_preds[:, 6].cpu().detach().numpy()
            # print(f"predicted values : {predicted_values}")
            mae = mean_absolute_error(true_values, predicted_values)
            
        # print(f"MAE : {mae}")
        
        mae = mae/len(self.val_loader)
        distance = distance / len(self.val_loader)

        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.global_acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.global_pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.global_rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.global_f1]}
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
        #print(pandas_data)

        self.wandb.log(data={ "%02d_val_loss" % (self.id) : total_loss})
            
        # print(f"Global iter {t}: Validation loss: {avg_loss}")
        # print(f"distance: {distance}")
        
        return total_loss, distance, pandas_data, mae
        
    def l1_distance_loss(self, prediction, target):
        loss = np.abs(prediction - target)
        return np.mean(loss)
        
    def evaluate_model(self):
        self.local_model
        total_loss=0.0

        informativeness_gt = []
        informativeness_pred = []
        for i, vdata in enumerate(self.val_loader):
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            total_loss += loss.item()
            
            # print(y_preds[:, :6].shape, information.shape)

            self.acc[0].update(y_preds[:, :6], information.to(self.device))
            self.pre[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.rec[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.f1[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.conf[0].update(y_preds[:, :6], information.to(self.device))
            
            self.distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())
            """for gt, y_pred in zip(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy()):
                informativeness_gt.append(int(gt))
                informativeness_pred.append(int(y_pred))"""
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

        #print(informativeness_gt)
        #print(informativeness_pred)
        #precision = precision_score(informativeness_gt, informativeness_pred, labels = np.arange(7))
        #print(f"precision : {precision}")
        #input("press")
        self.distance = self.distance / len(self.val_loader)

        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.f1]}
       
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}

        #print(pandas_data)

        avg_loss = total_loss / len(self.val_loader)
        
    def train(self):
        
        self.local_model.train()
        # print(self.local_iters)
        
        for iter in range(self.local_iters):
            mae = 0
            for ib, batch in enumerate(self.train_loader):
                features, additional_information, information, informativeness, sharingOwner, sharingOthers = batch
                self.optimizer.zero_grad()
                y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                loss.backward()
                self.optimizer.step()

                self.wandb.log(data={"%02d_train_loss" % (self.id) : loss/len(self.train_loader)})
                # print(f"Epoch : {iter} Training loss: {loss.item()}")
                # self.distance = 0.0
                
            self.evaluate_model()

        # self.distance = 0.0
    
                 

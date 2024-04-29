import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,  TensorDataset
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import torchvision.models as models

import copy
import pandas as pd
import numpy as np
from src.Optimizer.Optimizer import Fedmem
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from dipa2.ML_baseline.Study2TransMLP.inference_dataset import ImageMaskDataset
from src.TrainModels.trainmodels import PrivacyModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError
from sklearn.metrics import mean_absolute_error
import wandb
import sys

class User():

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
        self.country = "japan"
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
        self.exchange_model = copy.deepcopy(self.local_model)

        image_size = (224, 224)
        feature_folder = self.current_directory + '/object_features/resnet50/'

        num_rows = len(self.mega_table)
        train_size = int(0.8 * num_rows)
        test_size = num_rows - train_size
        self.train_samples = train_size
        self.test_samples = test_size
        self.samples = train_size + test_size
        # Split the dataframe into two
        train_df = self.mega_table.sample(n=train_size, random_state=0)
        val_df = self.mega_table.drop(train_df.index)

        train_dataset = ImageMaskDataset(train_df, feature_folder, self.input_channel, image_size, flip = True)
        val_dataset = ImageMaskDataset(val_df, feature_folder, self.input_channel, image_size)    

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, generator=torch.Generator(device='cuda'), shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), generator=torch.Generator(device='cuda'))
        self.trainloaderfull = DataLoader(train_dataset, batch_size=len(train_dataset), generator=torch.Generator(device='cuda'))
        
        # self.optimizer = Fedmem(self.local_model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate)
        self.exchange_optimizer= torch.optim.Adam(self.exchange_model.parameters(), lr=self.learning_rate)
        
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
 

        
    def set_parameters(self, global_model):
        for param, glob_param in zip(self.local_model.parameters(), global_model):
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
        avg_loss=0.0
        distance = 0.0
        mae=0.0
        for i, vdata in enumerate(self.trainloaderfull):
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            avg_loss += loss.item()/len(features)
            
            
            self.global_acc[0].update(y_preds[:, :6], information.to(self.device))
            self.global_pre[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.global_rec[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.global_f1[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to(self.device))
            self.global_conf[0].update(y_preds[:, :6], information.to(self.device))
            
            
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

            true_values = informativeness.cpu().detach().numpy()
            predicted_values = y_preds[:, 6].cpu().detach().numpy()
            mae += mean_absolute_error(true_values, predicted_values)/len(features)
            distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())/len(features)
            
        
        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.global_acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.global_pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.global_rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.global_f1]}
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
        #print(pandas_data)
        

        # avg_loss = total_loss / len(self.val_loader)
        # print(f"Global iter {t}: Validation loss: {avg_loss}")
        # print(f"distance: {distance}")
        
        self.wandb.log(data={ "%02d_train_loss" % int(self.id) : avg_loss})
        self.wandb.log(data={ "%02d_train_mae" % int((self.id)) : mae})
        """self.wandb.log(data={ "%02d_train_Accuracy" % int(self.id) : pandas_data['Accuracy']})
        self.wandb.log(data={ "%02d_train_precision" % int((self.id)) : pandas_data['Precision']})
        self.wandb.log(data={ "%02d_train_Recall" % int((self.id)) : pandas_data['Recall']})
        self.wandb.log(data={ "%02d_train_f1" % int((self.id)) : pandas_data['f1']})
        """
        return avg_loss, distance, pandas_data, mae
    


    def test(self, global_model, t):
        # Set the model to evaluation mode
        self.local_model.eval()
        self.update_parameters(global_model)
        avg_loss=0.0
        distance = 0.0
        mae = 0.0
        for i, vdata in enumerate(self.val_loader):
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            avg_loss += loss.item()/len(features)
            
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

            true_values = informativeness.cpu().detach().numpy()
            predicted_values = y_preds[:, 6].cpu().detach().numpy()
            mae += mean_absolute_error(true_values, predicted_values)/len(features)
            distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())/len(features)
            
        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.global_acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.global_pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.global_rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.global_f1]}
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
       
        
        self.wandb.log(data={ "%02d_val_loss" % int(self.id) : avg_loss})
        self.wandb.log(data={ "%02d_val_mae" % int((self.id)) : mae})
        """self.wandb.log(data={ "%02d_val_Accuracy" % int(self.id) : pandas_data['Accuracy']})
        self.wandb.log(data={ "%02d_val_precision" % int((self.id)) : pandas_data['Precision']})
        self.wandb.log(data={ "%02d_val_Recall" % int((self.id)) : pandas_data['Recall']})
        self.wandb.log(data={ "%02d_val_f1" % int((self.id)) : pandas_data['f1']})
        """
        return avg_loss, distance, pandas_data, mae
    
        
    def l1_distance_loss(self, prediction, target):
        loss = np.abs(prediction - target)
        return np.mean(loss)
        
    def evaluate_model(self):
        self.local_model.eval()
        avg_loss=0.0
        mae=0.0
        distance=0.0
        for i, vdata in enumerate(self.val_loader):
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            avg_loss += loss.item()/len(features)
            
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

            true_values = informativeness.cpu().detach().numpy()
            predicted_values = y_preds[:, 6].cpu().detach().numpy()
            mae += mean_absolute_error(true_values, predicted_values)/len(features)
            distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())/len(features)
            
        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.f1]}
       
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}

            
    def train(self):
        # print(f"user id : {self.id}")
        
        self.local_model.train()
        for iter in range(self.local_iters):
            mae = 0
            for batch in self.train_loader:
                features, additional_information, information, informativeness, sharingOwner, sharingOthers = batch
                self.optimizer.zero_grad()
                y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                loss.backward()
                self.optimizer.step()
               # print(f"User id : {self.id} :::: During Individual training Epoch : {iter} ::: Training loss: {loss.item()}")
                
            self.evaluate_model()

           
    def exchange_parameters(self, rl_model):
        #print(self.exchange_model.parameters())
        # print(rl_model)
        for param, rl_param in zip(self.exchange_model.parameters(), rl_model):
            #print(f"param :", param.data)
            #print(f"rl_param :", rl_param.data)
            param.data = rl_param.data.clone()
    
    
    def transfer_model(self, rl_user):
        for rl_param, param in zip(rl_user, self.exchange_model.parameters()):
            rl_param.data = param.data.clone()
           # rl_param.grad.data = param.grad.data.clone()
            # print(f"rl_param :", rl_param.data)
            # print(f"exchange_param :", param.data)
            

    def exchange_train(self, exchange_dict_users):
        for rl_user in exchange_dict_users:
            # print(f"low resource user id : {rl_user.id} and exchanged with user id : {self.id}")
            #print(rl_user.local_model)
            self.exchange_parameters(rl_user.local_model.parameters())
            
            self.exchange_model.train()
            for iter in range(self.local_iters):
                mae = 0
                for batch in self.train_loader:
                    features, additional_information, information, informativeness, sharingOwner, sharingOthers = batch
                    self.exchange_optimizer.zero_grad()
                    y_preds = self.exchange_model(features.to(self.device), additional_information.to(self.device))
                    criterion = self.exchange_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                    criterion.backward()
                    self.exchange_optimizer.step()
                    # print(f"RL_user : {rl_user.id} and RF_user : {self.id} During exchange Epoch : {iter} Training loss : {criterion.item()}")
                    
            self.transfer_model(rl_user.local_model.parameters())
    


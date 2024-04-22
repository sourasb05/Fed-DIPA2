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

        self.acc, self.pre, self.rec, self.f1, self.conf = [], [], [], [], []
        
        self.optimizer= torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate)
        
        
        
        
        
        

        
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
        self.eval_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.eval_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.test_loader)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        cm = confusion_matrix(y_true,y_pred)  
        
        file_cm = "cm_user_" + str(self.id) + "_GR_" + str(t)
        #print(file)
       
        directory_name = "global" 
        cm_df = pd.DataFrame(cm)

        if not os.path.exists(self.current_directory + "/results/confusion_matrix/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/confusion_matrix/"+ directory_name)
        
        cm_df.to_csv(self.current_directory + "/results/confusion_matrix/"+ directory_name + "/" + file_cm + ".csv", index=False)

        # print(f"local model saved at global round :{t} local round :{iter}")       
        return accuracy, validation_loss, precision, recall, f1, cm

    def test_local(self, t):
        self.best_local_model.eval()
        loss = 0
    
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.best_local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.test_loader)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        cm = confusion_matrix(y_true,y_pred)
        self.list_accuracy.append([accuracy, t])
        self.list_f1.append([f1, t]) 
        self.list_val_loss.append([validation_loss,t])       
        return accuracy, validation_loss, precision, recall, f1, cm


    def train_error_and_loss(self, global_model):
      
        # Set the model to evaluation mode
        self.eval_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.trainloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.eval_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.trainloaderfull)
        # precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
                
        return accuracy, train_loss

    def train_error_and_loss_local(self):
        self.best_local_model.eval()
        loss = 0
    
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.trainloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.trainloaderfull)
        # precision = precision_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
                
        return accuracy, train_loss

        
#    @staticmethod
#    def model_exists():
#        return os.path.exists(os.path.join("models", "server" + ".pt"))
    
    def l1_distance_loss(prediction, target):
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
            """print(y_preds[:, :6].shape, information.shape)

            self.acc[0].update(y_preds[:, :6], information.to('cuda'))
            self.pre[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
            self.rec[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
            self.f1[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
            self.conf[0].update(y_preds[:, :6], information.to('cuda'))
            """
            # distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())

            """self.acc[1].update(y_preds[:, 7:14], sharingOwner.to('cuda'))
            self.pre[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
            self.rec[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
            self.f1[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
            self.conf[1].update(y_preds[:, 7:14], sharingOwner.to('cuda'))

            self.acc[2].update(y_preds[:, 14:21], sharingOthers.to('cuda'))
            self.pre[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
            self.rec[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
            self.f1[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
            self.conf[2].update(y_preds[:, 14:21], sharingOthers.to('cuda'))

            """
        #  distance = distance / len(self.val_loader)

        """pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.f1]}
        """
        # print(pandas_data)
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation loss: {avg_loss}")


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
                print(f"Epoch : {iter} Training loss: {loss.item()}")
                self.evaluate_model()

        """    
        trainer = pl.Trainer(accelerator='gpu', devices=[0],logger=wandb_logger, 
        auto_lr_find=True, max_epochs = 100, callbacks=[checkpoint_callback])

        lr_finder = trainer.tuner.lr_find(self.local_model, self.train_loader)

        self.local_model.hparams.learning_rate = lr_finder.suggestion()
        print(f'lr auto: {lr_finder.suggestion()}')
        trainer.fit(self.local_model, train_dataloaders = self.train_loader, val_dataloaders = self.val_loader) 
        
        
        

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
        self.distance = 0.0
        """
                

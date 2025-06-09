import torch
import copy
import wandb
import sys
import math
import os
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset, random_split,  TensorDataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from dipa2.ML_baseline.Study2TransMLP.inference_dataset import ImageMaskDataset
from src.TrainModels.trainmodels import PrivacyModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError
from sklearn.metrics import mean_absolute_error
from src.utils.results_utils import CalculateMetrics, InformativenessMetrics

class User():

    def __init__(self,device, args, id, exp_no, current_directory, wandb):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device
        self.wandb = wandb
        self.id = id  # integer
        self.batch_size = args.batch_size
        self.exp_no = exp_no
        self.current_directory = current_directory
        self.num_glob_iters = args.num_global_iters
        self.delta=args.delta    
        self.kappa=args.kappa    
        self.lamda=args.lamda_sim_sta  ##tradeoff between similarity and stability
        self.send_to_server=0
        self.flag=0
        self.cluster_number = args.num_teams
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
        #print(str(self.id))
        self.mega_table = pd.read_csv(current_directory + '/feature_clients/annotations_annotator' + str(self.id) + '.csv')
        # print(self.mega_table)
       
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

        image_size = (224, 224)

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

        train_df.to_csv("%s/train_%d.csv" % (dataset_files_dir, int(self.id)), index=False)
        val_df.to_csv("%s/val_%d.csv" % (dataset_files_dir, int(self.id)), index=False)
        test_df.to_csv("%s/test_%d.csv" % (dataset_files_dir, int(self.id)), index=False)

        if not args.test:
            train_dataset = ImageMaskDataset(train_df, args.model_name, self.input_channel, image_size, flip = True)
            val_dataset = ImageMaskDataset(val_df, args.model_name, self.input_channel, image_size)
            #print(len(val_dataset))
            #print(len(train_dataset))
            #input("press")
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, generator=torch.Generator(device='cuda'), shuffle=True)
            self.trainloaderfull = DataLoader(train_dataset, batch_size=len(train_dataset), generator=torch.Generator(device='cuda'), shuffle=True)
            self.val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'), batch_size=len(val_dataset))

        test_dataset = ImageMaskDataset(test_df, args.model_name, self.input_channel, image_size)
        self.test_loader = DataLoader(test_dataset, generator=torch.Generator(device='cuda'), batch_size=16) #len(test_dataset))
        # Dataset Allocation ends

        self.local_model = PrivacyModel(input_dim=self.input_dim,
                                        max_bboxes=test_dataset.max_bboxes,
                                        features_dim=test_dataset.features_dim).to(self.device)
        if not args.test:
            self.exchange_model = copy.deepcopy(self.local_model)
            self.old_model = copy.deepcopy(self.local_model)
            # self.optimizer = Fedmem(self.local_model.parameters(), lr=self.learning_rate)
            self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate)
            self.exchange_optimizer= torch.optim.Adam(self.exchange_model.parameters(), lr=self.learning_rate)

        self.train_samples = train_size
        self.val_samples = val_size
        self.samples = train_size + val_size

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
        
        self.val_round_result_dict = {}
        self.val_cluster_round_result_dict = {}
        self.val_global_round_result_dict = {}
        
        self.train_round_result_dict = {}
        self.train_cluster_round_result_dict = {}
        self.train_global_round_result_dict = {}
        
        self.test_round_result_dict = {}
        self.test_cluster_round_result_dict = {}
        self.test_global_round_result_dict = {}
        

        self.minimum_test_loss = 10000000.0

        if args.test:
            self.model_status = self.load_model()

    def load_model(self):
        # models_dir = "./models/dynamic_FedDcprivacy/global_model/delta_1.0_kappa_1.0_GE_25_LE_2"
        # model_state_dict = torch.load(os.path.join(models_dir, "delta_0.1_kappa_0.1", "server_checkpoint_GR19.pt"))["model_state_dict"]

        # models_dir = "./models/dynamic_FedDcprivacy/local_model/" + str(self.id) + "/delta_1.0_kappa_1.0_lambda_0.3_GE_5_LE_10"
        #####  Cluster ablation
        # models_dir = f"./models/dynamic_FedDcprivacy/local_model/cluster_{self.cluster_number}/{self.id}/delta_1.0_kappa_1.0_lambda_0.3_GE_5_LE_10"
        # kappa delta ablation
        models_dir = f"./models/dynamic_FedDcprivacy/local_model/cluster_{self.cluster_number}/{self.id}/delta_1.0_kappa_1.0_lambda_0.5_GE_50_LE_1"
        # models_dir = "./models/FedMEM/local_model/" + str(71) + "/delta_1.0_kappa_1.0_lambda_0.5_GE_50_LE_1"
        
        # model_path = os.path.join(models_dir, str(self.id), "delta_%s_kappa_%s" % (self.delta, self.kappa), "local_checkpoint_GR19.pt")
        model_path = os.path.join(models_dir,  "best_local_checkpoint.pt")
        # model_path = os.path.join(models_dir,  "local_checkpoint_GR29.pt")

        
        if os.path.exists(model_path):         
            model_state_dict = torch.load(model_path)["model_state_dict"]
            self.local_model.load_state_dict(model_state_dict)
            self.local_model.eval()
            return True
        return False

    def set_parameters_old(self):
        for old_param, new_param in zip(self.old_model.parameters(), self.local_model.parameters()):
            old_param.data = new_param.data.clone()

        
    def set_parameters(self, global_model):
        for param, glob_param in zip(self.local_model.parameters(), global_model):
            param.data = glob_param.data.clone()
            # print(f"user {self.id} parameters : {param.data}")
            
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
            mae = mean_absolute_error(true_values, predicted_values)
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
        self.wandb.log(data={ "%02d_train_Accuracy" % int(self.id) : pandas_data['Accuracy'][0]})
        self.wandb.log(data={ "%02d_train_precision" % int((self.id)) : pandas_data['Precision'][0]})
        self.wandb.log(data={ "%02d_train_Recall" % int((self.id)) : pandas_data['Recall'][0]})
        self.wandb.log(data={ "%02d_train_f1" % int((self.id)) : pandas_data['f1'][0]})
        
        return avg_loss, distance, pandas_data, mae
    


    def test(self, global_model=None, t=0):
        # Set the model to evaluation mode
        self.local_model.eval()
        if global_model != None:
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
            # print(f"true :, {true_values}")
            # print(f"predicted :, {predicted_values}")
            # print(f"user id: {self.id}")
            
            mae = mean_absolute_error(true_values, predicted_values)
            distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())/len(features)
            
        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in self.global_acc], 
                    'Precision' : [i.compute().detach().cpu().numpy() for i in self.global_pre], 
                    'Recall': [i.compute().detach().cpu().numpy() for i in self.global_rec], 
                    'f1': [i.compute().detach().cpu().numpy() for i in self.global_f1]}
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
       
        
        
        self.wandb.log(data={ "%02d_val_loss" % int(self.id) : avg_loss})
        self.wandb.log(data={ "%02d_val_mae" % int((self.id)) : mae})
        self.wandb.log(data={ "%02d_val_Accuracy" % int(self.id) : pandas_data['Accuracy'][0]})
        self.wandb.log(data={ "%02d_val_precision" % int((self.id)) : pandas_data['Precision'][0]})
        self.wandb.log(data={ "%02d_val_Recall" % int((self.id)) : pandas_data['Recall'][0]})
        self.wandb.log(data={ "%02d_val_f1" % int((self.id)) : pandas_data['f1'][0]})
    
        return avg_loss, distance, pandas_data, mae
    
        
    def l1_distance_loss(self, prediction, target):
        loss = np.abs(prediction - target)
        return np.mean(loss)
    

    def evaluate_rl_model(self, t, rl_user):
        rl_user.local_model.eval()
        avg_loss=0.0
        mae=0.0
        distance=0.0
        for i, vdata in enumerate(rl_user.val_loader):
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = rl_user.local_model(features.to(rl_user.device), additional_information.to(rl_user.device))
            loss = rl_user.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            
            avg_loss += loss.item()/len(features)

        self.save_rl_model(t, avg_loss, rl_user)

    def save_rl_model(self, glob_iter, current_loss, rl_user):
      
        model_dir = os.path.join( rl_user.current_directory,
                                 "models",
                                 rl_user.algorithm,
                                 "local_model",
                                 str(rl_user.id),
                                 "cluster_" + str(rl_user.cluster_number)
                                 )

        model_filename = f"delta_{rl_user.delta}_kappa_{rl_user.kappa}_lambda_{rl_user.lamda}_GE_{rl_user.num_glob_iters}_LE_{rl_user.local_iters}"

        model_path = os.path.join(model_dir, model_filename)


        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        if glob_iter == rl_user.num_glob_iters-1:
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': rl_user.local_model.state_dict(),
                        'loss': rl_user.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "local_checkpoint_GR" + str(glob_iter) + ".pt"))
            
        if current_loss < rl_user.minimum_test_loss:
            if self.flag == 0:
                self.send_to_server+=1
                self.flag = 1
            self.minimum_test_loss = current_loss
            
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.local_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "best_local_checkpoint" + ".pt"))
        
    def evaluate_model(self, t, epoch):
        self.local_model.eval()
        avg_loss=0.0
        mae=0.0
        distance=0.0
        for i, vdata in enumerate(self.val_loader):
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            
            avg_loss += loss.item()/len(features)
            
            true_values = informativeness.cpu().detach().numpy()
            predicted_values = y_preds[:, 6].cpu().detach().numpy()
            mae = mean_absolute_error(true_values, predicted_values)
            distance += self.l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())/len(features)

        #print("avg_loss_outside_loop:",avg_loss)
        #input("press")
        
        self.save_model(t,epoch, avg_loss)

    def save_model(self, glob_iter, epoch, current_loss):
            
        model_path = ( f"{self.current_directory}/models/{self.algorithm}/local_model/CFedDC_rl1_C{self.cluster_number}/{self.id}/delta_{self.delta}_kappa_{self.kappa}_lambda_{self.lamda}_GE_{self.num_glob_iters}_LE_{self.local_iters}"
)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        model_dir = os.path.join( self.current_directory, 
                                 "models",
                                 self.algorithm,
                                 "local_model",
                                 str(self.id),
                                 "cluster_" + str(self.cluster_number)
            )

        model_subdir = f"delta_{self.delta}_kappa_{self.kappa}_lambda_{self.lamda}_GE_{self.num_glob_iters}_LE_{self.local_iters}"
        model_path = os.path.join(model_dir, model_subdir)

        if not os.path.exists(model_path):
            os.makedirs(model_path)



        if glob_iter == self.num_glob_iters-1:
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.local_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "local_checkpoint_GR" + str(glob_iter) + ".pt"))
            
        if current_loss < self.minimum_test_loss:
            if self.flag == 0:
                self.send_to_server+=1
                self.flag = 1
            self.minimum_test_loss = current_loss

            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.local_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "best_local_checkpoint" + ".pt"))
    
    def calculate_similarity(self, cluster_model, exchange=None):
        l_similarity = 0.0
        if exchange == None:
            for global_param, curr_local_param in zip(cluster_model, self.local_model.parameters()):
                l_similarity +=  0.5*torch.norm(curr_local_param - global_param) ** 2
        else:
            for global_param, curr_local_param in zip(cluster_model, self.exchange_model.parameters()):
                l_similarity +=  0.5*torch.norm(curr_local_param - global_param) ** 2

        
        return l_similarity

    def calculate_stability(self,rl=None):
        l_stability = 0.0
        if rl == None:
            for prev_param, curr_param in zip(self.old_model.parameters(), self.local_model.parameters()):
                l_stability += 0.5* torch.norm(prev_param - curr_param) ** 2
        else:
            for prev_param, curr_param in zip(rl.local_model.parameters(), self.exchange_model.parameters()):
                l_stability += 0.5* torch.norm(prev_param - curr_param) ** 2


        return l_stability 

    # Extracting parameters and flattening them into feature vectors
    def get_model_parameters(self, user):
        split_model = nn.Sequential(*list(user.local_model.children())[-10:])
        
        params = []
        
        for param in user.local_model.parameters():
            params.extend(param.detach().cpu().numpy().flatten())  # Flatten each parameter tensor
        
        return np.array(params)

    def train(self,cluster_model,t):
        # print(f"user id : {self.id}")
        self.set_parameters(cluster_model)
        self.local_model.train()
        for iter in range(self.local_iters):
            for batch in self.train_loader:
                features, additional_information, information, informativeness, sharingOwner, sharingOthers = batch
                self.optimizer.zero_grad()
                y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                loss1 = self.calculate_similarity(cluster_model)
                loss2 = self.calculate_stability()
                self.set_parameters_old()
                loss = loss + self.lamda*loss1 + (1-self.lamda)*loss2
                loss.backward()
                self.optimizer.step()
                
            self.evaluate_model(t, iter)

           
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

        

    def exchange_train(self, rl_user, cluster_model, t):

        self.exchange_parameters(rl_user.local_model.parameters())
            
        self.exchange_model.train()
        for iter in range(self.local_iters):
            mae = 0
            for batch in self.train_loader:
                features, additional_information, information, informativeness, sharingOwner, sharingOthers = batch
                self.exchange_optimizer.zero_grad()
                y_preds = self.exchange_model(features.to(self.device), additional_information.to(self.device))
                criterion = self.exchange_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                loss1 = self.calculate_similarity(cluster_model, exchange=1)
                loss2 = self.calculate_stability(rl=rl_user)
                
                loss = criterion + self.lamda*loss1 + (1-self.lamda)*loss2

                criterion.backward()
                self.exchange_optimizer.step()

                # print(f"RL_user : {rl_user.id} and RF_user : {self.id} During exchange Epoch : {iter} Training loss : {criterion.item()}")
                    
        self.transfer_model(rl_user.local_model.parameters())
        self.evaluate_rl_model(t, rl_user)
    
    
    def test_eval(self):
        self.local_model.eval()

        results = []
        for i, vdata in enumerate(self.test_loader):
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
        return results
    

    def test_global_model_test(self, model):
        if model != None:
            self.set_parameters(model)
        self.local_model.eval()
        
        acc = [Accuracy(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(output_channel.items())]

        results = []
        for i, vdata in enumerate(self.test_loader):
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])

            # print(y_preds[:, :6].shape, information.shape)
            acc[0].update(y_preds[:, :6], information.to(self.device))
            acc[1].update(y_preds[:, 7:14], sharingOwner.to(self.device))
            acc[2].update(y_preds[:, 14:21], sharingOthers.to(self.device))
           
        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in acc], 
                    }
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
        print(pandas_data)
        
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Precision, Recall, F1Score]
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.__name__] = [metric(task="multilabel",
                                                    num_labels=output_dim,
                                                    threshold = threshold,
                                                    average=average_method,
                                                    ignore_index = output_dim - 1) \
                                                    for i, (output_name, output_dim) in enumerate(output_channel.items())]
        informativeness_scores = [[], []]

        for result in results:
            information, informativeness, sharingOwner, sharingOthers, y_preds = result
            gt = [information, sharingOwner, sharingOthers]
            output_dims = output_channel.values()
            for o, (output_dim, gt) in enumerate(zip(output_dims, gt)):
                start_dim = o*(output_dim)
                end_dim = o*(output_dim)+output_dim
                for metric_name in metrics_data.keys():
                    metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())
        results_data = {}
        for metric_name in metrics_data.keys():
            results_data[metric_name] = [i.compute().detach().cpu().numpy() for i in metrics_data[metric_name]]

        results_clean_dict = { key: [float(val) for val in value] for key, value in results_data.items()}

        result_dict = {**pandas_data, **results_clean_dict}

        print(result_dict)
        

        # for i, k in enumerate(output_channel.keys()):
        #     for metric, values in results_data.items():
        #         print("%.02f " % values[i], end="")

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(informativeness_scores[0], informativeness_scores[1])
        # print("%.02f %.02f %.02f %.02f %.02f" % (info_prec, info_rec, info_f1, info_cmae, info_mae))

        # Check if it's the first round (i.e., the result_round_dict is empty)
        if not self.test_round_result_dict:
        # Initialize by converting each list into a list of lists
            self.test_round_result_dict = {k: [v] for k, v in result_dict.items()}
            self.test_round_result_dict.update({ 'info_prec': [info_prec],
                                            'info_rec': [info_rec],
                                            'info_f1': [info_f1],
                                            'info_cmae': [info_cmae],
                                            'info_mae': [info_mae]})
        else:
        # Append new values to the existing lists
            for k in result_dict:
                self.test_round_result_dict[k].append(result_dict[k])
            self.test_round_result_dict['info_prec'].append(info_prec)
            self.test_round_result_dict['info_rec'].append(info_rec)
            self.test_round_result_dict['info_f1'].append(info_f1)
            self.test_round_result_dict['info_cmae'].append(info_cmae)
            self.test_round_result_dict['info_mae'].append(info_mae)


        return info_prec, info_rec, info_f1, info_cmae, info_mae, result_dict


    def test_global_model_val(self, model):
        if model != None:
            self.set_parameters(model)
        self.local_model.eval()

        results = []

        acc = [Accuracy(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(output_channel.items())]
        avg_loss = 0.0
        for i, vdata in enumerate(self.val_loader):
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            avg_loss += loss.item()/len(features)

            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
            
            # print(y_preds[:, :6].shape, information.shape)
            acc[0].update(y_preds[:, :6], information.to(self.device))
            acc[1].update(y_preds[:, 7:14], sharingOwner.to(self.device))
            acc[2].update(y_preds[:, 14:21], sharingOthers.to(self.device))
           
        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in acc], 
                    }
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
        print(pandas_data)
        
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Precision, Recall, F1Score]
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.__name__] = [metric(task="multilabel",
                                                    num_labels=output_dim,
                                                    threshold = threshold,
                                                    average=average_method,
                                                    ignore_index = output_dim - 1) \
                                                    for i, (output_name, output_dim) in enumerate(output_channel.items())]
        informativeness_scores = [[], []]

        for result in results:
            information, informativeness, sharingOwner, sharingOthers, y_preds = result
            gt = [information, sharingOwner, sharingOthers]
            output_dims = output_channel.values()
            for o, (output_dim, gt) in enumerate(zip(output_dims, gt)):
                start_dim = o*(output_dim)
                end_dim = o*(output_dim)+output_dim
                for metric_name in metrics_data.keys():
                    metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())
        results_data = {}
        for metric_name in metrics_data.keys():
            results_data[metric_name] = [i.compute().detach().cpu().numpy() for i in metrics_data[metric_name]]

        results_clean_dict = { key: [float(val) for val in value] for key, value in results_data.items()}

        result_dict = {**pandas_data, **results_clean_dict}

        print(result_dict)
        

        # for i, k in enumerate(output_channel.keys()):
        #     for metric, values in results_data.items():
        #         print("%.02f " % values[i], end="")

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(informativeness_scores[0], informativeness_scores[1])
        # print("%.02f %.02f %.02f %.02f %.02f" % (info_prec, info_rec, info_f1, info_cmae, info_mae))

        # Check if it's the first round (i.e., the result_round_dict is empty)
        if not self.val_global_round_result_dict:
        # Initialize by converting each list into a list of lists
            self.val_global_round_result_dict = {k: [v] for k, v in result_dict.items()}
            self.val_global_round_result_dict.update({ 'info_prec': [info_prec],
                                            'info_rec': [info_rec],
                                            'info_f1': [info_f1],
                                            'info_cmae': [info_cmae],
                                            'info_mae': [info_mae]})
        else:
        # Append new values to the existing lists
            for k in result_dict:
                self.val_global_round_result_dict[k].append(result_dict[k])
            self.val_global_round_result_dict['info_prec'].append(info_prec)
            self.val_global_round_result_dict['info_rec'].append(info_rec)
            self.val_global_round_result_dict['info_f1'].append(info_f1)
            self.val_global_round_result_dict['info_cmae'].append(info_cmae)
            self.val_global_round_result_dict['info_mae'].append(info_mae)


        return avg_loss, info_prec, info_rec, info_f1, info_cmae, info_mae, result_dict
    
    def test_cluster_model_val(self, model):
        if model != None:
            self.set_parameters(model)
        self.local_model.eval()

        results = []

        acc = [Accuracy(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
                for i, (output_name, output_dim) in enumerate(output_channel.items())]
        
        avg_loss=0.0
       
        for i, vdata in enumerate(self.val_loader):
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            avg_loss += loss.item()/len(features)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
            
            # print(y_preds[:, :6].shape, information.shape)
            acc[0].update(y_preds[:, :6], information.to(self.device))
            acc[1].update(y_preds[:, 7:14], sharingOwner.to(self.device))
            acc[2].update(y_preds[:, 14:21], sharingOthers.to(self.device))
           
        pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in acc], 
                    }
        
        pandas_data = {k: [float(v) for v in values] for k, values in pandas_data.items()}
        print(pandas_data)
        
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Precision, Recall, F1Score]
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.__name__] = [metric(task="multilabel",
                                                    num_labels=output_dim,
                                                    threshold = threshold,
                                                    average=average_method,
                                                    ignore_index = output_dim - 1) \
                                                    for i, (output_name, output_dim) in enumerate(output_channel.items())]
        informativeness_scores = [[], []]

        for result in results:
            information, informativeness, sharingOwner, sharingOthers, y_preds = result
            gt = [information, sharingOwner, sharingOthers]
            output_dims = output_channel.values()
            for o, (output_dim, gt) in enumerate(zip(output_dims, gt)):
                start_dim = o*(output_dim)
                end_dim = o*(output_dim)+output_dim
                for metric_name in metrics_data.keys():
                    metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())
        results_data = {}
        for metric_name in metrics_data.keys():
            results_data[metric_name] = [i.compute().detach().cpu().numpy() for i in metrics_data[metric_name]]

        results_clean_dict = { key: [float(val) for val in value] for key, value in results_data.items()}

        result_dict = {**pandas_data, **results_clean_dict}

        # print(result_dict)
        

        # for i, k in enumerate(output_channel.keys()):
        #     for metric, values in results_data.items():
        #         print("%.02f " % values[i], end="")

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(informativeness_scores[0], informativeness_scores[1])
        # print("%.02f %.02f %.02f %.02f %.02f" % (info_prec, info_rec, info_f1, info_cmae, info_mae))

        # Check if it's the first round (i.e., the result_round_dict is empty)
        if not self.val_global_round_result_dict:
        # Initialize by converting each list into a list of lists
            self.val_cluster_round_result_dict = {k: [v] for k, v in result_dict.items()}
            self.val_cluster_round_result_dict.update({ 'info_prec': [info_prec],
                                            'info_rec': [info_rec],
                                            'info_f1': [info_f1],
                                            'info_cmae': [info_cmae],
                                            'info_mae': [info_mae]})
        else:
        # Append new values to the existing lists
            for k in result_dict:
                self.val_cluster_round_result_dict[k].append(result_dict[k])
            self.val_cluster_round_result_dict['info_prec'].append(info_prec)
            self.val_cluster_round_result_dict['info_rec'].append(info_rec)
            self.val_cluster_round_result_dict['info_f1'].append(info_f1)
            self.val_cluster_round_result_dict['info_cmae'].append(info_cmae)
            self.val_cluster_round_result_dict['info_mae'].append(info_mae)


        return avg_loss, info_prec, info_rec, info_f1, info_cmae, info_mae, result_dict


    
    def test_local_model_test(self):
       
        self.local_model.eval()

        results = []
        for i, vdata in enumerate(self.test_loader):
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])

            
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Accuracy, Precision, Recall, F1Score]
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.__name__] = [metric(task="multilabel",
                                                    num_labels=output_dim,
                                                    threshold = threshold,
                                                    average=average_method,
                                                    ignore_index = output_dim - 1) \
                                                    for i, (output_name, output_dim) in enumerate(output_channel.items())]
        informativeness_scores = [[], []]

        for result in results:
            information, informativeness, sharingOwner, sharingOthers, y_preds = result
            gt = [information, sharingOwner, sharingOthers]
            output_dims = output_channel.values()
            for o, (output_dim, gt) in enumerate(zip(output_dims, gt)):
                start_dim = o*(output_dim)
                end_dim = o*(output_dim)+output_dim
                for metric_name in metrics_data.keys():
                    metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())
        results_data = {}
        for metric_name in metrics_data.keys():
            results_data[metric_name] = [i.compute().detach().cpu().numpy() for i in metrics_data[metric_name]]

        result_dict = { key: [float(val) for val in value] for key, value in results_data.items()}

        
        # print(result_dict)
        

        # for i, k in enumerate(output_channel.keys()):
        #     for metric, values in results_data.items():
        #         print("%.02f " % values[i], end="")

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(informativeness_scores[0], informativeness_scores[1])
        print("User ID: %s %.02f %.02f %.02f %.02f %.02f" % (self.id, info_prec, info_rec, info_f1, info_cmae, info_mae))

        # Check if it's the first round (i.e., the result_round_dict is empty)
        if not self.test_round_result_dict:
        # Initialize by converting each list into a list of lists
            self.test_round_result_dict = {k: [v] for k, v in result_dict.items()}
            self.test_round_result_dict.update({ 'info_prec': [info_prec],
                                            'info_rec': [info_rec],
                                            'info_f1': [info_f1],
                                            'info_cmae': [info_cmae],
                                            'info_mae': [info_mae]})
        else:
        # Append new values to the existing lists
            for k in result_dict:
                self.test_round_result_dict[k].append(result_dict[k])
            self.test_round_result_dict['info_prec'].append(info_prec)
            self.test_round_result_dict['info_rec'].append(info_rec)
            self.test_round_result_dict['info_f1'].append(info_f1)
            self.test_round_result_dict['info_cmae'].append(info_cmae)
            self.test_round_result_dict['info_mae'].append(info_mae)


        return info_prec, info_rec, info_f1, info_cmae, info_mae, result_dict


    def test_local_model_val(self):
      
        self.local_model.eval()

        results = []
        for i, vdata in enumerate(self.val_loader):
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
        
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Accuracy, Precision, Recall, F1Score]
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.__name__] = [metric(task="multilabel",
                                                    num_labels=output_dim,
                                                    threshold = threshold,
                                                    average=average_method,
                                                    ignore_index = output_dim - 1) \
                                                    for i, (output_name, output_dim) in enumerate(output_channel.items())]
        informativeness_scores = [[], []]

        for result in results:
            information, informativeness, sharingOwner, sharingOthers, y_preds = result
            gt = [information, sharingOwner, sharingOthers]
            output_dims = output_channel.values()
            for o, (output_dim, gt) in enumerate(zip(output_dims, gt)):
                start_dim = o*(output_dim)
                end_dim = o*(output_dim)+output_dim
                for metric_name in metrics_data.keys():
                    metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())
        results_data = {}
        for metric_name in metrics_data.keys():
            results_data[metric_name] = [i.compute().detach().cpu().numpy() for i in metrics_data[metric_name]]

        result_dict = { key: [float(val) for val in value] for key, value in results_data.items()}

        
        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(informativeness_scores[0], informativeness_scores[1])
        print("User ID: %s %.02f %.02f %.02f %.02f %.02f" % (self.id, info_prec, info_rec, info_f1, info_cmae, info_mae))

        # Check if it's the first round (i.e., the result_round_dict is empty)
        if not self.val_round_result_dict:
        # Initialize by converting each list into a list of lists
            self.val_round_result_dict = {k: [v] for k, v in result_dict.items()}
            self.val_round_result_dict.update({ 'info_prec': [info_prec],
                                            'info_rec': [info_rec],
                                            'info_f1': [info_f1],
                                            'info_cmae': [info_cmae],
                                            'info_mae': [info_mae]})
        else:
        # Append new values to the existing lists
            for k in result_dict:
                self.val_round_result_dict[k].append(result_dict[k])
            self.val_round_result_dict['info_prec'].append(info_prec)
            self.val_round_result_dict['info_rec'].append(info_rec)
            self.val_round_result_dict['info_f1'].append(info_f1)
            self.val_round_result_dict['info_cmae'].append(info_cmae)
            self.val_round_result_dict['info_mae'].append(info_mae)


        return info_prec, info_rec, info_f1, info_cmae, info_mae, result_dict
    
    def test_model(self, model):

        self.set_parameters(model)
        self.local_model.eval()

        results = []
        for i, vdata in enumerate(self.test_loader):
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])

        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Precision, Recall, F1Score]
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.__name__] = [metric(task="multilabel",
                                                    num_labels=output_dim,
                                                    threshold = threshold,
                                                    average=average_method,
                                                    ignore_index = output_dim - 1) \
                                                    for i, (output_name, output_dim) in enumerate(output_channel.items())]
        informativeness_scores = [[], []]

        for result in results:
            information, informativeness, sharingOwner, sharingOthers, y_preds = result
            gt = [information, sharingOwner, sharingOthers]
            output_dims = output_channel.values()
            for o, (output_dim, gt) in enumerate(zip(output_dims, gt)):
                start_dim = o*(output_dim)
                end_dim = o*(output_dim)+output_dim
                for metric_name in metrics_data.keys():
                    metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())
        results_data = {}
        for metric_name in metrics_data.keys():
            results_data[metric_name] = [i.compute().detach().cpu().numpy() for i in metrics_data[metric_name]]

        # for i, k in enumerate(output_channel.keys()):
        #     for metric, values in results_data.items():
        #         print("%.02f " % values[i], end="")

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(informativeness_scores[0], informativeness_scores[1])
        # print("%.02f %.02f %.02f %.02f %.02f" % (info_prec, info_rec, info_f1, info_cmae, info_mae))

        return info_cmae
    

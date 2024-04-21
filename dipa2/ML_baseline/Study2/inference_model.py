import torch
from torch import nn
from torchvision.models import VGG16_Weights, ResNet18_Weights, ResNet50_Weights, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError
import json

class BaseModel(pl.LightningModule):
    def __init__(self, input_dim, learning_rate = 1e-4, dropout_prob=0.2):
        ## output_channel: key: output_name value: output_dim
        super().__init__()
        self.learning_rate = learning_rate
        
        # mobilenet v3 
        # self.net = torch.hub.load('pytorch/vision:v0.14.1', 'mobilenet_v3_large', pretrained=MobileNet_V3_Large_Weights.DEFAULT)
        # self.net.classifier[3] = nn.Identity()
        # w0 = self.net.features[0][0].weight.data.clone()
        # self.net.features[0][0] = nn.Conv2d(3 + input_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.net.features[0][0].weight.data[:,:3,:,:] = w0
        # self.fc1 = nn.Linear(1280, 256)
        #resnet 50
        self.net = torch.hub.load('pytorch/vision:v0.14.1', 'resnet50', pretrained=ResNet50_Weights.DEFAULT)
        self.net.fc = nn.Identity()
        w0 = self.net.conv1.weight.data.clone()
        self.net.conv1 = nn.Conv2d(3 + input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.conv1.weight.data[:,:3,:,:] = w0
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 21)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.act = nn.SiLU()
        self.reg_loss = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        #for information type
        self.entropy_loss1 = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight = torch.tensor([1.,1.,1.,1.,1.,0.]))
        self.entropy_loss2 = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight = torch.tensor([1.,1.,1.,1.,1.,1.,0.]))

    def forward(self, image, mask):
        x = self.net(torch.cat((image, mask), dim = 1))
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_loss(self, image, mask, information, informativeness, sharingOwner, sharingOthers, text='train'):
        y_preds = self(image, mask)
        #0 ~5: type 6: informativeness 7~13: sharingOwners 14~20: sharingOthers
        TypeLoss = self.entropy_loss1(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        informativenessLosses = self.reg_loss(y_preds[:,6] * 100, informativeness.type(torch.FloatTensor).to('cuda') * 100) 
        sharingOwenerLoss = self.entropy_loss2(y_preds[:,7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        sharingOthersLoss = self.entropy_loss2(y_preds[:,14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        loss = TypeLoss + informativenessLosses + sharingOwenerLoss + sharingOthersLoss
        self.log(f'{text} loss', loss)
        self.log(f'{text} type loss', TypeLoss)
        self.log(f'{text} informativeness loss', informativenessLosses)
        self.log(f'{text} sharingOwnerloss', sharingOwenerLoss)
        self.log(f'{text} sharingOthersloss', sharingOthersLoss)
        self.save_metrics(y_preds, information, informativeness, sharingOwner, sharingOthers, text=text)
        return loss
    def training_step(self, batch, batch_idx):
        image, mask, information, informativeness, sharingOwner, sharingOthers = batch
        loss = self.get_loss(image, mask, information, informativeness, sharingOwner, sharingOthers)
    
        return loss
    
    def validation_step (self, val_batch, batch_idx):
        image, mask, information, informativeness, sharingOwner, sharingOthers = val_batch
        vloss = self.get_loss(image, mask, information, informativeness, sharingOwner, sharingOthers, text='val')
        return vloss  
    
    def save_metrics(self, y_preds, information, informativeness, sharingOwner, sharingOthers, text='val', average_method = 'weighted', threshold = 0.5):
        def l1_distance_loss(prediction, target):
            loss = np.abs(prediction - target)
            return np.mean(loss)
        
        accuracy = Accuracy(task="multilabel", num_labels=6, threshold = threshold, average=average_method, ignore_index = 5)
        precision = Precision(task="multilabel", num_labels=6, threshold = threshold,average=average_method, ignore_index = 5)
        recall = Recall(task="multilabel", num_labels=6,threshold = threshold,average=average_method, ignore_index = 5)
        f1score = F1Score(task="multilabel", num_labels=6, threshold = threshold,average=average_method, ignore_index = 5)

        accuracy.update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        precision.update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        recall.update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        f1score.update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))

        self.log(f"{text}/acc for information type", accuracy.compute())
        self.log(f"{text}/pre for information type", precision.compute())
        self.log(f"{text}/rec for information type", recall.compute())
        self.log(f"{text}/f1 for information type", f1score.compute())
        
        distance = l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())
        self.log(f"{text}/distance for informativeness", distance)

        accuracy = Accuracy(task="multilabel", num_labels=7, threshold = threshold,average=average_method, ignore_index = 6)
        precision = Precision(task="multilabel", num_labels=7, threshold = threshold,average=average_method, ignore_index = 6)
        recall = Recall(task="multilabel", num_labels=7, threshold = threshold,average=average_method, ignore_index = 6)
        f1score = F1Score(task="multilabel", num_labels=7, threshold = threshold,average=average_method, ignore_index = 6)

        accuracy.update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        precision.update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        recall.update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        f1score.update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))

        self.log(f"{text}/acc for sharing as owner", accuracy.compute())
        self.log(f"{text}/pre for sharing as owner", precision.compute())
        self.log(f"{text}/rec for sharing as owner", recall.compute())
        self.log(f"{text}/f1 for sharing as owner", f1score.compute())
        
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1score.reset()

        accuracy.update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        precision.update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        recall.update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        f1score.update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))

        self.log(f"{text}/acc for sharing by others", accuracy.compute())
        self.log(f"{text}/pre for sharing by others", precision.compute())
        self.log(f"{text}/rec for sharing by others", recall.compute())
        self.log(f"{text}/f1 for sharing by others", f1score.compute())
       
   

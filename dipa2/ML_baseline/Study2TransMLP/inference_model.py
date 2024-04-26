import torch
from torch import nn
from torchvision.models import VGG16_Weights, ResNet18_Weights, ResNet50_Weights, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError
import json
import sys

import math
from torchvision.ops import roi_pool, roi_align
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
from torch.optim.lr_scheduler import ExponentialLR

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer' 
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, nhead)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        #src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

class BaseModel(pl.LightningModule):
    def __init__(self, input_dim, learning_rate = 0.01, dropout_prob=0.2):
        ## output_channel: key: output_name value: output_dim
        super().__init__()
        self.learning_rate = learning_rate

        self.features_dim = (2048, 7, 7)
        self.max_bboxes = 32
        self.bb_features_channels = self.features_dim[0]
        self.num_additional_input = input_dim
        self.final_features_dim = 1024

        # Transformer Config
        self.transformer_input_len = self.max_bboxes
        self.transformer_latent_dim = self.bb_features_channels 
        self.transformer_hidden_dim = 512
        #self.transformer_nhead = int(self.final_features_dim/self.transformer_input_len)
        self.transformer_nhead = 32
        self.transformer_nlayers = 2

        self.transformer = TransformerModel(ntoken  = self.transformer_input_len,
                                            d_model = self.transformer_latent_dim,
                                            nhead   = self.transformer_nhead,
                                            d_hid   = self.transformer_hidden_dim,
                                            nlayers = self.transformer_nlayers,
                                            dropout = 0.5)
        
        #self.transformer_out_avgpool = nn.AvgPool1d(self.transformer_input_len, stride=1)
        self.transformer_out_mlp = nn.Linear(self.transformer_nhead *
                                             self.transformer_input_len,
                                             self.final_features_dim)

        self.mlp_fc1 = nn.Linear(self.num_additional_input, 256)
        self.mlp_fc2 = nn.Linear(256, self.final_features_dim)

        # self.fusion_fc1 = nn.Linear(self.final_features_dim, 512)
        self.fusion_fc1 = nn.Linear(self.final_features_dim*2, 512)
        self.fusion_fc2 = nn.Linear(512, 256)
        self.fusion_fc3 = nn.Linear(256, 21)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # mobilenet v3 
        # self.net = torch.hub.load('pytorch/vision:v0.14.1', 'mobilenet_v3_large', pretrained=MobileNet_V3_Large_Weights.DEFAULT)
        # self.net.classifier[3] = nn.Identity()
        # w0 = self.net.features[0][0].weight.data.clone()
        # self.net.features[0][0] = nn.Conv2d(3 + input_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.net.features[0][0].weight.data[:,:3,:,:] = w0
        # self.fc1 = nn.Linear(1280, 256)

        # original resnet 50
        # self.net = torch.hub.load('pytorch/vision:v0.14.1', 'resnet50', pretrained=ResNet50_Weights.DEFAULT)
        # self.net.fc = nn.Identity()
        # w0 = self.net.conv1.weight.data.clone()
        # self.net.conv1 = nn.Conv2d(3 + input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.net.conv1.weight.data[:,:3,:,:] = w0

        # self.fc1 = nn.Linear(2048, 256)
        # self.fc2 = nn.Linear(256, 21)
        # self.dropout = nn.Dropout(p=dropout_prob)
        self.act = nn.SiLU()

        self.reg_loss = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        #for information type
        self.entropy_loss1 = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight = torch.tensor([1.,1.,1.,1.,1.,0.]))
        self.entropy_loss2 = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight = torch.tensor([1.,1.,1.,1.,1.,1.,0.]))

    def forward(self, bb_features, additional_input):

        transformer_out = self.transformer(bb_features)
        transformer_out = torch.flatten(transformer_out, start_dim=1)
        #image_features = self.transformer_out_avgpool(transformer_out)
        image_features = self.transformer_out_mlp(transformer_out)
        image_features = transformer_out

        ai_features = self.mlp_fc1(additional_input)
        ai_features = self.relu(ai_features)
        ai_features = self.mlp_fc2(ai_features)

        x = torch.cat((image_features, ai_features), dim=1)

        x = self.relu(x)
        x = self.fusion_fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fusion_fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fusion_fc3(x)

        return x
            
        # x = self.net(torch.cat((image, mask), dim = 1))
        # x = self.dropout(x)
        # x = self.act(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        # return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.999)
        return {
                 "optimizer": optimizer,
                 "lr_scheduler": lr_scheduler
               }

    def get_loss(self, bb_features, additional_input, information, informativeness, sharingOwner, sharingOthers, text='train'):
        y_preds = self(bb_features, additional_input)

        N = len(y_preds)

        #0 ~5: type 6: informativeness 7~13: sharingOwners 14~20: sharingOthers
        TypeLoss = self.entropy_loss1(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        informativenessLosses = self.reg_loss(y_preds[:,6] * 100, informativeness.type(torch.FloatTensor).to('cuda') * 100) 
        sharingOwenerLoss = self.entropy_loss2(y_preds[:,7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        sharingOthersLoss = self.entropy_loss2(y_preds[:,14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))

        TypeLoss /= N
        informativenessLosses /= N
        sharingOwenerLoss /= N
        sharingOthersLoss /= N

        loss = TypeLoss + informativenessLosses + sharingOwenerLoss + sharingOthersLoss
        self.log(f'{text} loss', loss)
        self.log(f'{text} type loss', TypeLoss)
        self.log(f'{text} informativeness loss', informativenessLosses)
        self.log(f'{text} sharingOwnerloss', sharingOwenerLoss)
        self.log(f'{text} sharingOthersloss', sharingOthersLoss)
        self.save_metrics(y_preds, information, informativeness, sharingOwner, sharingOthers, text=text)
        return loss
    
    def training_step(self, batch, batch_idx):
        bb_features, additional_input, information, informativeness, sharingOwner, sharingOthers = batch
        loss = self.get_loss(bb_features, additional_input, information, informativeness, sharingOwner, sharingOthers)
    
        return loss
    
    def validation_step (self, val_batch, batch_idx):
        bb_features, additional_input, information, informativeness, sharingOwner, sharingOthers = val_batch
        vloss = self.get_loss(bb_features, additional_input, information, informativeness, sharingOwner, sharingOthers, text='val')
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
       
   

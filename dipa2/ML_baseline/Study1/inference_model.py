import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_Weights, ResNet18_Weights, ResNet50_Weights, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics
import json

class BaseModel(pl.LightningModule):
    def __init__(self, input_dim = 0, learning_rate = 1e-4, dropout_prob=0.2):
        ## output_channel: key: output_name value: output_dim
        super().__init__()
        self.learning_rate = learning_rate
        self.net = torch.hub.load('pytorch/vision:v0.14.1', 'resnet50', pretrained=ResNet50_Weights.DEFAULT)
        self.net.fc = nn.Identity()
        w0 = self.net.conv1.weight.data.clone()
        self.net.conv1 = nn.Conv2d(3 + input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.conv1.weight.data[:,:3,:,:] = w0
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 23)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.act = nn.ReLU()
        self.entropy_loss1 = nn.CrossEntropyLoss()

    def forward(self, image):
        x = self.net(image)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x) # Apply sigmoid function to convert output into a probability.
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer        
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float()) 
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log('test_loss', loss)
        return logits

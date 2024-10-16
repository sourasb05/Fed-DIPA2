import torch.nn as nn
import torch
import torchvision
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import torch
from torch import nn
# from torchvision.models import VGG16_Weights, ResNet18_Weights, ResNet50_Weights, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor

import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError
import json
import sys


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



class PrivacyModel(nn.Module):
    def __init__(self, input_dim, learning_rate = 0.01, dropout_prob=0.2):
        ## output_channel: key: output_name value: output_dim
        super().__init__()
        self.learning_rate = learning_rate

        self.features_dim = (2048, 7, 7)
        self.max_bboxes = 32
        self.bb_features_channels = self.features_dim[0]
        self.num_additional_input = input_dim
        self.final_features_dim = 512
        self.object_mlp_out_dim = 256

        # Transformer Config
        self.transformer_input_len = self.max_bboxes
        self.transformer_latent_dim = self.object_mlp_out_dim
        self.transformer_hidden_dim = 64
        self.transformer_nhead = int(self.final_features_dim/self.max_bboxes)
        self.transformer_nlayers = 1

        self.object_mlp = nn.Linear(self.bb_features_channels, self.object_mlp_out_dim)

        self.transformer = TransformerModel(ntoken  = self.transformer_input_len,
                                            d_model = self.transformer_latent_dim,
                                            nhead   = self.transformer_nhead,
                                            d_hid   = self.transformer_hidden_dim,
                                            nlayers = self.transformer_nlayers,
                                            dropout = 0.5)

        self.mlp_fc1 = nn.Linear(self.num_additional_input, 256)
        self.mlp_fc2 = nn.Linear(256, self.final_features_dim)

        # self.fusion_fc1 = nn.Linear(self.final_features_dim, 512)
        self.fusion_fc1 = nn.Linear(self.final_features_dim*2, 512)
        self.fusion_fc2 = nn.Linear(512, 256)
        self.fusion_fc3 = nn.Linear(256, 21)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.act = nn.SiLU()

        self.reg_loss = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        #for information type
        self.entropy_loss1 = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight = torch.tensor([1.,1.,1.,1.,1.,0.]))
        self.entropy_loss2 = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight = torch.tensor([1.,1.,1.,1.,1.,1.,0.]))

    def forward(self, object_features, additional_input):

        B, L, C = object_features.shape
        object_mlp_out = self.object_mlp(object_features.view(B*L, C))
        object_mlp_out = self.relu(object_mlp_out)
        object_mlp_out = object_mlp_out.view(B, L, self.object_mlp_out_dim)

        transformer_out = self.transformer(object_mlp_out)
        transformer_out = torch.flatten(transformer_out, start_dim=1)
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

    # def __init__(self, input_dim, learning_rate = 0.01, dropout_prob=0.2):
    #     ## output_channel: key: output_name value: output_dim
    #     super().__init__()
    #     self.learning_rate = learning_rate

    #     self.features_dim = (2048, 7, 7)
    #     self.max_bboxes = 32
    #     self.bb_features_channels = self.features_dim[0]
    #     self.num_additional_input = input_dim
    #     self.final_features_dim = 1024

    #     # Transformer Config
    #     self.transformer_input_len = self.max_bboxes
    #     self.transformer_latent_dim = self.bb_features_channels 
    #     self.transformer_hidden_dim = 512
    #     self.transformer_nhead = 32
    #     self.transformer_nlayers = 2

    #     self.transformer = TransformerModel(ntoken  = self.transformer_input_len,
    #                                         d_model = self.transformer_latent_dim,
    #                                         nhead   = self.transformer_nhead,
    #                                         d_hid   = self.transformer_hidden_dim,
    #                                         nlayers = self.transformer_nlayers,
    #                                         dropout = 0.5)
        
    #     self.transformer_out_mlp = nn.Linear(self.transformer_nhead *
    #                                          self.transformer_input_len,
    #                                          self.final_features_dim)

    #     self.mlp_fc1 = nn.Linear(self.num_additional_input, 256)
    #     self.mlp_fc2 = nn.Linear(256, self.final_features_dim)

    #     self.fusion_fc1 = nn.Linear(self.final_features_dim*2, 512)
    #     self.fusion_fc2 = nn.Linear(512, 256)
    #     self.fusion_fc3 = nn.Linear(256, 21)

    #     self.relu = nn.ReLU()
    #     self.dropout = nn.Dropout(p=0.2)

    #     self.act = nn.SiLU()

    #     self.reg_loss = nn.L1Loss()
    #     self.sigmoid = nn.Sigmoid()
    #     #for information type
    #     self.entropy_loss1 = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight = torch.tensor([1.,1.,1.,1.,1.,0.]))
    #     self.entropy_loss2 = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight = torch.tensor([1.,1.,1.,1.,1.,1.,0.]))

    # def forward(self, bb_features, additional_input):
    #     # print(bb_features)
    #     # print(f"additional input : {additional_input}")
    #     # sys.exit()
    #     transformer_out = self.transformer(bb_features)
    #     transformer_out = torch.flatten(transformer_out, start_dim=1)
    #     image_features = self.transformer_out_mlp(transformer_out)
    #     image_features = transformer_out

    #     ai_features = self.mlp_fc1(additional_input)
    #     ai_features = self.relu(ai_features)
    #     ai_features = self.mlp_fc2(ai_features)

    #     x = torch.cat((image_features, ai_features), dim=1)

    #     x = self.relu(x)
    #     x = self.fusion_fc1(x)
    #     x = self.act(x)
    #     x = self.dropout(x)
    #     x = self.fusion_fc2(x)
    #     x = self.act(x)
    #     x = self.dropout(x)
    #     x = self.fusion_fc3(x)

    #     return x
    
    def compute_loss(self, y_preds, information, informativeness, sharingOwner, sharingOthers):
        TypeLoss = self.entropy_loss1(y_preds[:, :6], information.type(torch.FloatTensor).cuda())
        informativenessLosses = self.reg_loss(y_preds[:,6] * 100, informativeness.type(torch.FloatTensor).cuda() * 100)
        sharingOwnerLoss = self.entropy_loss2(y_preds[:,7:14], sharingOwner.type(torch.FloatTensor).cuda())
        sharingOthersLoss = self.entropy_loss2(y_preds[:,14:21], sharingOthers.type(torch.FloatTensor).cuda())
        total_loss = TypeLoss + informativenessLosses + sharingOwnerLoss + sharingOthersLoss
        return total_loss



#=================Old code using image=================
class BaseModel(nn.Module):
    def __init__(self, input_dim, learning_rate=1e-4, dropout_prob=0.2):
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate
        self.net = resnet50(pretrained=True)
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
        self.entropy_loss1 = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor([1.,1.,1.,1.,1.,0.]))
        self.entropy_loss2 = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor([1.,1.,1.,1.,1.,1.,0.]))

    def forward(self, image, mask):
        x = self.net(torch.cat((image, mask), dim=1))
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def compute_loss(self, y_preds, information, informativeness, sharingOwner, sharingOthers):
        TypeLoss = self.entropy_loss1(y_preds[:, :6], information.type(torch.FloatTensor).cuda())
        informativenessLosses = self.reg_loss(y_preds[:,6] * 100, informativeness.type(torch.FloatTensor).cuda() * 100)
        sharingOwnerLoss = self.entropy_loss2(y_preds[:,7:14], sharingOwner.type(torch.FloatTensor).cuda())
        sharingOthersLoss = self.entropy_loss2(y_preds[:,14:21], sharingOthers.type(torch.FloatTensor).cuda())
        total_loss = TypeLoss + informativenessLosses + sharingOwnerLoss + sharingOthersLoss
        return total_loss


#==============================================================================================
class ResNet50TL(nn.Module):

    def __init__(self, n_class, fc2=True):
        super(ResNet50TL, self).__init__() 
        resnet = torchvision.models.resnet50(pretrained=True)
        self.avgpool = nn.Sequential(list(resnet.children())[-2])
        # Define linear layers and ReLU activation
        self.fc1 = nn.Linear(2048, 512)  # Assuming the output of avgpool is [batch_size, 2048, 1, 1]
        self.dropout = nn.Dropout(0.5)  # Dropout p=0.5
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, n_class)
        
    def forward(self, x):
        features_1d = self.avgpool(x)
        features_1d = torch.flatten(features_1d, 1)  # Flatten the features
        x = self.fc1(features_1d)
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

#==============================

class ResNet50FC(nn.Module):

    def __init__(self):
        super(ResNet50FC, self).__init__() 
        resnet = torchvision.models.resnet50(pretrained=True)
        self.avgpool = nn.Sequential(list(resnet.children())[-2])
        # Define linear layers and ReLU activation
        self.fc1 = nn.Linear(2048, 512)  # Assuming the output of avgpool is [batch_size, 2048, 1, 1]
        self.dropout = nn.Dropout(0.5)  # Dropout p=0.5
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        features_1d = self.avgpool(x)
        features_1d = torch.flatten(features_1d, 1)  # Flatten the features
        x = self.fc1(features_1d)
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        return x



class CEMNet(nn.Module):
    def __init__(self,n_class, mlp_input_size, mlp_hidden_size, mlp_output_size, checkpoint_path):
        super(CEMNet, self).__init__()
        # Initialize the ResNet50TL model for image feature extraction
        # self.cnn = ResNet50TL(n_class=n_class, fc2=False)
        """
        load the best client model.
        """
        if checkpoint_path:
            self.cnn = torch.load(checkpoint_path)
            if hasattr(self.cnn, 'fc2'):
                delattr(self.cnn, 'fc2')
       # print(self.cnn)
       #  input("press")
        self.cnn.eval()
        # Define MLP for contextual information
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.Sigmoid(),
            nn.Linear(mlp_hidden_size, mlp_output_size),
            nn.Softmax(dim=1),
        )

        # Define FC Layer after concatenation
        # Assuming n_class is the output size from CNN, and mlp_output_size is the output size from MLP
        self.fc = nn.Linear(512 + mlp_output_size, n_class)  # Assuming the final score is a single scalar

    def forward(self, image, context_info):
        # Forward pass through CNN
        # print(image)
        # print(f"self.cnn : {self.cnn}")
        image_features = self.cnn(image)
        # print(image_features)
        # Forward pass through MLP
        
        context_features = self.mlp(context_info)

        # Concatenate features
        combined_features = torch.cat((image_features, context_features), dim=1)

        # Forward pass through final FC layer to get the event memory score
        event_memory_score = self.fc(combined_features)

        return event_memory_score







#==============================================================

class VGG16FC(nn.Module):
    def __init__(self):
        super(VGG16FC, self).__init__()
        model = models.vgg16(pretrained=True)
        self.core_cnn = nn.Sequential(*list(model.features.children())[:-7])  # to relu5_3`
        self.D=512
        return

    def forward(self, x):
        x = self.core_cnn(x)
        return x

class ResNet18FC(nn.Module):
    def __init__(self):
        super(ResNet18FC, self).__init__()
        self.core_cnn = models.resnet18(pretrained=True)
        self.D=256
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


class ResNet50FC(nn.Module):
    def __init__(self):
        super(ResNet50FC, self).__init__()
        self.core_cnn = models.resnet50(pretrained=True)
        self.D = 1024
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


class ResNet101FC(nn.Module):
    def __init__(self):
        super(ResNet101FC, self).__init__()
        self.core_cnn = models.resnet101(pretrained=True)
        self.D = 1024
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


#==============================================================
# Direct ResNet50 memorability estimation - no attention or RNN
class ResNet50FT(nn.Module):
    def __init__(self):
        super(ResNet50FT, self).__init__()
        self.core_cnn = models.resnet50(pretrained=True)
        self.avgpool = nn.AvgPool2d(7)
        expansion = 2
        self.fc = nn.Linear(512 * expansion, 10)
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        #x = self.core_cnn.layer3(x)
        # x = self.core_cnn.layer4(x)
        
        x = self.avgpool(x)
       
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        # output_seq = x.unsqueeze(1)
        output = F.log_softmax(x, dim=1)
        

        # output = None
        # alphas = None
        return output # , output_seq, alphas


#===============================================================================================


#===============================================================================================
class AMemNetModel(nn.Module):

    def __init__(self, core_cnn, hps, a_res = 14, a_vec_size=512,num_features=27):
        super(AMemNetModel, self).__init__()

        self.hps = hps
        self.use_attention = hps.use_attention
        #self.force_distribute_attention = hps.force_distribute_attention
        self.with_bn = True

        self.a_vec_size = a_vec_size    # D
        self.a_vec_num = a_res * a_res  # L

        self.seq_len = hps.seq_steps
        self.lstm_input_size = self.a_vec_size
        self.lstm_hidden_size = 1024  # H Also LSTM output
        self.lstm_layers = 1

        self.core_cnn = core_cnn

        self.inconv = nn.Conv2d(in_channels=core_cnn.D, out_channels=a_vec_size, kernel_size=(1,1), stride=1, padding=0, bias=True)
        if self.with_bn: self.bn1 = nn.BatchNorm2d(a_vec_size)


        # Layers for the h and c LSTM states
        self.hs1 = nn.Linear(in_features=self.a_vec_size, out_features=self.lstm_hidden_size)
        self.hc1 = nn.Linear(in_features=self.a_vec_size, out_features=self.lstm_hidden_size)

        # e layers
        self.e1 = nn.Linear(in_features=self.a_vec_size, out_features=self.a_vec_size, bias=False)

        # Context layers
        self.eh1 = nn.Linear(in_features=self.lstm_hidden_size, out_features=self.a_vec_num)
        self.eh3 = nn.Linear(in_features=self.a_vec_size, out_features=1, bias=False)

        # LSTM
        self.rnn = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
                        num_layers=self.lstm_layers, dropout=0.5, bidirectional=False)

        # Regression Network
        self.regnet1 = nn.Linear(in_features=self.lstm_hidden_size, out_features=512)
        #self.regnet4 = nn.Linear(in_features=self.regnet1.out_features, out_features=1)
        self.regnet4 = nn.Linear(1024, out_features=10)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.drop80 = nn.Dropout(0.80)

        if hps.torch_version_major == 0 and hps.torch_version_minor < 3:
            self.softmax = nn.Softmax()
        else:
            self.softmax = nn.Softmax(dim=1)
        ################################MLP#######################
        self.mlp_fc1=nn.Linear(num_features, 512)

    def forward(self, x,fea):

        if not self.use_attention:
            self.alpha = torch.Tensor(x.size(0), self.a_vec_num)
            self.alpha = Variable(self.alpha)
            if self.hps.use_cuda:
                self.alpha = self.alpha.cuda()

            nn.init.constant(self.alpha, 1)
            self.alpha = self.alpha / self.a_vec_num
       # print(x.shape)
        x = self.core_cnn(x)

        x = self.inconv(x)
        if self.with_bn: x = self.bn1(x)
        x = self.relu(x) # -> [B, D, Ly, Lx] [B, 512, 14, 14]
        x = self.drop80(x)

        a = x.view(x.size(0), self.a_vec_size, self.a_vec_num)  # [B, D, L]

        # Extract the annotation vector
        # Mean of each feature map
        af = a.mean(2) # [B, D]

        # Hidden states for the LSTM
        hs = self.hs1(af)  # [D->H]
        hs = self.tanh(hs)

        cs = self.hc1(af) # [D->H]
        cs = self.tanh(cs)

        e = a.transpose(2, 1).contiguous() # -> [B, L, D]
        e = e.view(-1, self.a_vec_size) # a=[B, L, D] -> (-> [B*L, D])
        e = self.e1(e) # [B*L, D] -> [B*L, D]
        e = self.relu(e)
        e = self.drop50(e)
        e = e.view(-1, self.a_vec_num, self.a_vec_size) # -> [B, L, D]
        e = e.transpose(2,1) # -> [B, D, L]

        # Execute the LSTM steps
        h = hs
        rnn_state = (hs.expand(self.lstm_layers, hs.size(0), hs.size(1)).contiguous(),
                     cs.expand(self.lstm_layers, cs.size(0), cs.size(1)).contiguous())

        steps = self.seq_len
        if steps == 0:
            steps = 1

        output_seq = [0] * steps
        alphas = [0] * steps

        for i in range(steps):

            if self.use_attention:

                # Dynamic part of the alpha map from the current hidden RNN state
                if 0:
                    eh = self.eh12(h)  # -> [H -> D]
                    eh = eh.view(-1, self.a_vec_size, 1) # [B, D, 1]
                    eh = e+eh # [B, D, L]  + [B, D, 1]  => adds the eh vec[D] to all positions [L] of the e tensor

                if 1:
                    eh = self.eh1(h)  # -> [H -> L]
                    eh = eh.view(-1, 1, self.a_vec_num)  # [B, 1, L]
                    eh = e+eh  # [B, D, L]  + [B, 1, L]

                eh = self.relu(eh)
                eh = self.drop50(eh)

                eh = eh.transpose(2, 1).contiguous()  # -> [B, L, D]
                eh = eh.view(-1, self.a_vec_size)  # -> [B*L, D]

                eh = self.eh3(eh)  # -> [B*L, 512] -> [B*L, 1]
                eh = eh.view(-1, self.a_vec_num)  # -> [B, L]


                alpha = self.softmax(eh) # -> [B, L]

            else:
                alpha = self.alpha

            alpha_a = alpha.view(alpha.size(0), self.a_vec_num, 1) # -> [B, L, 1]
            z = a.bmm(alpha_a) # ->[B, D, 1] scale the location feature vectors by the alpha mask and add them (matrix mul)
            # [D, L] * [L] = [D]

            z = z.view(z.size(0), self.a_vec_size)
            z = z.expand(1, z.size(0), z.size(1)) # Prepend a new, single dimension representing the sequence

            if self.seq_len == 0:
                z = z.squeeze(dim=0)
                h = self.drop50(z)

                out = self.regnet1(h)
                out = self.relu(out)
                out = self.drop50(out)
                out = self.regnet4(out)

                output_seq[0] = out
                alphas[0] = alpha.unsqueeze(1)

                break

            # Run RNN step
            self.rnn.flatten_parameters()
            h, rnn_state = self.rnn(z, rnn_state)
            h = h.squeeze(dim=0)  # remove the seqeunce dimension
            h = self.drop50(h)
            
            #print(fea.shape)
            fea1 = self.mlp_fc1(fea)
            out = self.regnet1(h)
            #print(fea.shape, out.shape)
            out = torch.cat((fea1, out), dim=1)

            out = self.relu(out)
            out = self.drop50(out)
            out = self.regnet4(out)
            #print(out)
            # Store the output and the attention mask
            ind = i
            out=out.unsqueeze(1)
            output_seq[ind] = out
            alphas[ind] = alpha.unsqueeze(1)


        output_seq = torch.cat(output_seq, 1)
        alphas = torch.cat(alphas, 1)
        #print(output_seq)
        output = None
        return output, output_seq, alphas

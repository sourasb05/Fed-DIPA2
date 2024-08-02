import torch
import copy
from client.UserBase import Base_user
import torch.nn as nn

class MOON_client(Base_user):

    def __init__(self, device, args, id, exp_no, current_directory, wandb):
        super().__init__(device, args, id, exp_no, current_directory, wandb)
    
        self.criterion = nn.CrossEntropyLoss()

    def initialize_previous_model(self):
        for prev_param, curr_param in zip(self.prev_model.parameters(), self.local_model.parameters()):
            prev_param.data = curr_param.data.clone()
            # prev_param.grad.data = curr_param.grad.data.clone()    

    def initialize_global_model(self, global_model_params):
        for prev_global_param, curr_global_param in zip(self.global_model.parameters(),  global_model_params):
            prev_global_param.data = curr_global_param.data.clone()
            #prev_global_param.grad.data = curr_global_param.grad.data.clone()          
    

    
    def train(self, global_model_param):
        self.initialize_previous_model()
        self.initialize_global_model(global_model_param)
        self.local_model.train()
        # print(self.local_iters)
        cos=torch.nn.CosineSimilarity(dim=-1)

        for iter in range(self.local_iters):
            mae = 0
            for ib, batch in enumerate(self.train_loader):
                features, additional_information, information, informativeness, sharingOwner, sharingOthers = batch
                self.optimizer.zero_grad()

                features.requires_grad = False
                
                _, pro1, y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
                _, pro2, _ = self.global_model(features.to(self.device), additional_information.to(self.device))

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                _, pro3, _ = self.prev_model(features.to(self.device), additional_information.to(self.device))
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                logits /= self.temperature
                labels = torch.zeros(features.size(0)).cuda().long()
                
                loss2 = self.mu * self.criterion(logits, labels)

                loss1 = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                
                loss = loss1 + loss2
                
                loss.backward()
                self.optimizer.step()

                self.wandb.log(data={"%02d_train_loss" % (self.id) : loss/len(self.train_loader)})
                # print(f"Epoch : {iter} Training loss: {loss.item()}")
                # self.distance = 0.0
                
            #self.evaluate_model()

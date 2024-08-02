
from client.UserBase import Base_user



class FedAvg_client(Base_user):
    def __init__(self,device, args, id, exp_no, current_directory, wandb):
        super().__init__(device, args, id, exp_no, current_directory, wandb)


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
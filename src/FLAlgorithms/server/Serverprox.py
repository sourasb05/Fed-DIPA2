import copy
from tqdm import trange
from tqdm import tqdm
import sys
from server.ServerBase import Base_server
from client.Userprox import FedProx_client


class FedProx_server(Base_server):
    def __init__(self,device, args, exp_no, current_directory):
        super().__init__(device, args, exp_no, current_directory)

        
        for i in trange(self.total_users, desc="Data distribution to clients"):
            user = FedProx_client(self.device, args, int(self.user_ids[i]), exp_no, current_directory, self.wandb)
            if user.valid: # Copy for all algorithms
                self.users.append(user)
                self.total_train_samples += user.train_samples
            
            self.total_users = len(self.users) 
            self.num_users = self.total_users * args.users_frac    #selected users
        
        
        #Create Global_model
        for user in self.users:
            self.data_frac.append(user.train_samples/self.total_train_samples)
        print(f"data available {self.data_frac}")
        self.global_model = copy.deepcopy(self.users[0].local_model)
        for param in self.global_model.parameters():
            param.data.zero_()

        print("Finished creating FedProx server.")
    
    def train(self):
        loss = []
        
        for glob_iter in trange(self.num_glob_iters, desc="Global Rounds"):
            self.send_parameters()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)
            #print(f"Exp no{self.exp_no} : users selected for global iteration {glob_iter} are : {list_user_id}")

            for user in self.selected_users:
                user.train(self.global_model.parameters())  # * user.train_samples

            self.aggregate_parameters()
            self.evaluate(glob_iter)
            self.save_model(glob_iter)
        self.save_results()

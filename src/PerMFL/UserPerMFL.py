import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.Optimizer.Optimizer import PerMFL
from tqdm import trange
import copy
import os


# Implementation for pFedMe clients

class UserPerMFL():
    def __init__(self, device, train_data, test_data, model, args, id):
        
        """
        device : Cuda Available
        id : user_id
        """

        self.device = device
        self.id = id

        
        """
        Data and batch sizes
        """
        
        
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size

        """
        Hyper-parameters
        """

        self.alpha = args.alpha
        self.beta = args.beta
        self.lamda = args.lamda
        self.eta = args.eta
        self.gamma = args.gamma
        self.local_epochs = args.local_epochs

        """
        DataLoaders for setting trainloader, testloaders, and produce iterables
        """

        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader = DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        """
        Models for Federated Learning in clients
        """
        self.model = copy.deepcopy(model)
        self.model_team = copy.deepcopy(model)
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.personalized_team_model = copy.deepcopy(list(self.model.parameters()))

        """
        Select Loss
        """
        self.loss = nn.CrossEntropyLoss()

        """
        Select optimizer
        """
        self.optimizer = PerMFL(self.model.parameters(), alpha=self.alpha, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]


    def set_parameters(self, team_model):
        for local_param, param in zip(self.model.parameters(), team_model.parameters()):
            local_param.data = param.data.clone()
        # self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()  ## to remove the require_grad and only fetch the tensor
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self, global_model):
        self.model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id, " + , Test Accuracy:", test_acc / int(y.shape[0]) )
            # print(self.id, " + , Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, loss, y.shape[0]

    def train_error_and_loss(self, global_model):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(self.local_model)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)

        self.update_parameters(self.local_model)
        return test_acc, loss, y.shape[0]

    def test_personalized_team_model(self, team_update):
        self.model.eval()
        test_acc = 0
        loss = 0
        # self.personalized_team_model = copy.deepcopy(list(team_update))
        self.update_parameters(team_update)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, loss, y.shape[0]

    def train_error_and_loss_personalized_team_model(self, team_update):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(team_update)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def train_error_and_loss_personalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
            # print("X :", len(X), "y :", len(y))

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
            # print("In exception X :", len(X), "y :", len(y))
        return (X.to(self.device), y.to(self.device))

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))

    def train(self, iters, team_model):
        team_model_list = copy.deepcopy(list(team_model))
        for iter in trange(iters): 
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.personalized_model_bar, _ = self.optimizer.step(team_model_list)

            for new_param, localweight in zip(self.personalized_model_bar, self.local_model):
                localweight.data = new_param.data
        
        self.update_parameters(self.local_model)



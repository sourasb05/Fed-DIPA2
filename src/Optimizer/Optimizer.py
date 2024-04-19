from torch.optim import Optimizer
import torch

class Fedmem(Optimizer):
    def __init__(self, params, alpha=0.05, eta=0.05):

        defaults = dict(alpha=alpha, eta=eta)
        super(Fedmem, self).__init__(params, defaults)

    def step(self, cluster_model, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            
            for p, cluster_weight in zip(group['params'],cluster_model):
                if p.grad is None:
                    continue

                p.data = p.data - group['alpha']*(p.grad.data +  group['eta']*(p.data - cluster_weight.data))
               
        return loss

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                if p.grad is None:
                    continue
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            print(type(weight_update))
            print(type(group['params']))
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']


class PerMFL(Optimizer):
    def __init__(self, params, alpha=0.01, lamda=0.1):
        # self.local_weight_updated = local_weight # w_i,K
        if alpha < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))
        if lamda < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))

        defaults = dict(alpha=alpha, lamda=lamda)
        super(PerMFL, self).__init__(params, defaults)

    def step(self, cluster_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = cluster_updated.copy()
        for group in self.param_groups:
            for theta, clusterweight in zip(group['params'], weight_update):
                theta.data = theta.data - group['alpha'] * theta.grad.data + group['lamda'] * (theta.data - clusterweight.data)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for theta, localweight in zip(group['params'], weight_update):
                theta.data = localweight.data
        # return  p.data
        return group['params']

import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import numpy as np
import time
from scipy.stats import qmc
import scipy.io

from .model import sequential_model

class slowly_varying_10(sequential_model):
    def __init__(self, layers):
        super().__init__(layers)


    def function(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)

        x.requires_grad = True

        epsilon = 1/10
        ax = 0.5*torch.sin(2*np.pi*x/epsilon) + torch.sin(x) + 2

        u = self.forward(x)

        u_x = autograd.grad(u,x,torch.ones(x.shape).to(self.device), retain_graph=True, create_graph=True)[0]
        u_xx = autograd.grad(u_x*ax, x, torch.ones(x.shape).to(self.device), retain_graph = True, create_graph = True)[0]

        f = u_xx + torch.sin(x)
        return f

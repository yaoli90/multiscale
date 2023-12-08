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

class diff_reac(sequential_model):
    def __init__(self, layers):
        super().__init__(layers)

    def function(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)

        g = x
        g.requires_grad = True

        u = self.forward(g)
        x_ = g[:,[0]]
        t_ = g[:,[1]]

        u_x_t = autograd.grad(u,g,torch.ones([x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        u_x = u_x_t[:,[0]]
        u_t = u_x_t[:,[1]]

        epsilon = 1/10.

        r = torch.cos(x_/epsilon)

        u_xx_xt = autograd.grad(u_x, g, torch.ones([x.shape[0], 1]).to(self.device), retain_graph = True, create_graph = True)[0]
        u_xx = u_xx_xt[:,[0]]

        f = u_t - 2*u_xx + 1/epsilon*r*u - torch.sin(2*np.pi*x_)
        return f

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

class elliptic_2d_non(sequential_model):
    def __init__(self, layers):
        super().__init__(layers)

    def function(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)

        g = x
        g.requires_grad = True

        u = self.forward(g)
        x_ = g[:,[0]]
        y_ = g[:,[1]]

        u_x_y = autograd.grad(u,g,torch.ones([x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        u_x = u_x_y[:,[0]]
        u_y = u_x_y[:,[1]]

        epsilon = 0.01

        a = 1 + .9*torch.sin(2*np.pi*x_/epsilon)*torch.cos(2*np.pi*y_/epsilon)

        u_xx_xy = autograd.grad((1+a*u**2)*u_x, g, torch.ones([x.shape[0], 1]).to(self.device), retain_graph = True, create_graph = True)[0]
        u_xx = u_xx_xy[:,[0]]

        u_yx_yy = autograd.grad((1+a*u**2)*u_y, g, torch.ones([x.shape[0], 1]).to(self.device), retain_graph = True, create_graph = True)[0]
        u_yy = u_yx_yy[:,[1]]



        f = u_xx + u_yy + 50
        return f

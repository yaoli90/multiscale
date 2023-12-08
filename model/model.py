import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import numpy as np
from scipy.stats import qmc


'''
    hypercube collocation point sampling
        input:
                bound: [[lb, ub], [lb, ub], ...]
                N: n samples
        output:
                x: with shape [N, dim]
'''
def latin_hypercube_sample(bound, N=1e3):
    bound = np.array(bound)
    dim = bound.shape[0]
    lb = bound[:,0]
    ub = bound[:,1]
    sampler = qmc.LatinHypercube(d=dim)
    x = lb + (ub-lb)*sampler.random(n=int(N))
    return x

'''
    hypercube collocation point sampling
        input:
                bound: [[lb, ub], [lb, ub], ...]
                N: n samples
                gt: groundtruth of the boundary
        output:
                x: with shape [N, dim]
                u: with shape [N, dim]
'''
def boundary_sample(bound, N=200, gt=lambda x:np.zeros((x.shape[0],1))):
    x = latin_hypercube_sample(bound, N)
    u = gt(x)
    return x,u


class sequential_model(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0
        self.bc_weight = 1
        self.fourier = None

        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    '''
    Forward Propagation
        input: x := [x, t]
        output: u(x,theta) '''
    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)

        if self.fourier is not None:
            x_f = [x]
            i = 0
            for n in self.fourier:
                for k in range(n):
                    x_f.append(torch.unsqueeze(torch.sin(k*np.pi*x[:,i]),1))
                    x_f.append(torch.unsqueeze(torch.cos(k*np.pi*x[:,i]),1))
                i =+ 1
            x = torch.cat(x_f, 1)

        for i in range(len(self.layers)-2):
            z = self.linears[i](x)
            x = self.activation(z)

        output = self.linears[-1](x)
        return output

    '''
    Model Residual
        input: x := [x, t]
        output: r(x,theta) '''
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



    def loss_BC(self, x_boundary, u_boundary):
        return self.loss_function(self.forward(x_boundary), u_boundary)

    def loss_PDE(self, f):
        loss_f = self.loss_function(f, torch.zeros(f.shape).to(self.device))
        return loss_f

    def loss(self, x_boundary, u_boundary, x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)
        if torch.is_tensor(x_boundary) != True:
            x_boundary = torch.from_numpy(x_boundary).float().to(self.device)
        if torch.is_tensor(u_boundary) != True:
            u_boundary = torch.from_numpy(u_boundary).float().to(self.device)

        f = self.function(x)

        loss_u = self.loss_BC(x_boundary, u_boundary)
        loss_f = self.loss_PDE(f)
        return self.bc_weight*loss_u + loss_f

    def train_model_adam(self, optimizer, x_boundary, u_boundary, x_train, n_epoch):
        while self.iter < n_epoch:
            optimizer.zero_grad()
            loss = self.loss(x_boundary, u_boundary, x_train)
            loss.backward()
            self.iter += 1
            if self.iter % 1000 == 0:
                print(self.iter, loss)
            optimizer.step()

    '''
    Test Model
        input: x, u
        output: rmse, u_pred '''
    def test(self, x, u):
        if torch.is_tensor(u) != True:
            u = torch.from_numpy(u).float().to(self.device)
        u_pred = self.forward(x)
        rmse = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)
        u_pred = u_pred.cpu().detach().numpy()
        return rmse, u_pred

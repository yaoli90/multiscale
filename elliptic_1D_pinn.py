import numpy as np
import torch
import pickle

from model import elliptic_1d, latin_hypercube_sample, boundary_sample, plot_1d

''' Settings and seeds '''
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

''' test dataset '''
x = np.linspace(np.pi/2048,np.pi*2047/2048,1024)
x_test = x.reshape(x.shape[0],1)
print("test dataset:", x_test.shape, '\n')

with open('groundtruth_generation/elliptic_1D_eps_1_8.pkl', 'rb') as file:
    u_test = pickle.load(file)
u_test = np.array(u_test).reshape(u_test.shape[0],1)

''' model '''
layers = np.array([1,40,40,40,40,40,40,40,1]) #7 hidden layers
PINN = elliptic_1d(layers)
PINN.to(PINN.device)
PINN.bc_weight = 200



def retrain(model, x_train, x_boundary, u_boundary, x_test, u_test, optimizer, epoch, idx=''):
    model.iter = 0
    model.train_model_adam(optimizer, x_boundary, u_boundary, x_train, epoch)

    f = np.abs(model.function(x_test).squeeze().cpu().detach().numpy())
    plot_1d(f, x_test, ylim=[-1,3], path="./elliptic_1D/res_pinn"+idx+".png")
    print('Test residual MSE:', (np.mean(np.square(f))))

    rmse, u_pred = model.test(x_test,u_test)
    print('Test RMSE:', rmse)
    plot_1d(u_pred, x_test, u_gt=u_test, ylim=[0,.6], path="./elliptic_1D/pre_pinn"+idx+".png")


N0 = 400

x_inner = latin_hypercube_sample([[0,np.pi]], N0)
x_l, u_l = boundary_sample([[0,0]], 1)
x_r, u_r = boundary_sample([[np.pi,np.pi]], 1)


optimizer = torch.optim.Adam(PINN.parameters(), lr=0.001)
retrain(PINN, np.vstack([x_inner, x_l, x_r]), np.vstack([x_l, x_r]), np.vstack([u_l, u_r]), x_test, u_test, \
    optimizer, 1e5, idx="1")

optimizer = torch.optim.Adam(PINN.parameters(), lr=0.0001)
retrain(PINN, np.vstack([x_inner, x_l, x_r]), np.vstack([x_l, x_r]), np.vstack([u_l, u_r]), x_test, u_test, \
    optimizer, 1e5, idx="2")

optimizer = torch.optim.Adam(PINN.parameters(), lr=0.00001)
retrain(PINN, np.vstack([x_inner, x_l, x_r]), np.vstack([x_l, x_r]), np.vstack([u_l, u_r]), x_test, u_test, \
    optimizer, 1e5, idx="3")

import numpy as np
import torch
import pickle

from model import elliptic_2d, latin_hypercube_sample, boundary_sample, plot_2d

''' Settings and seeds '''
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

''' test dataset '''
x = np.linspace(1/2048, 1-1/2048, 1024)
y = np.linspace(1/2048, 1-1/2048, 1024)
X, Y = np.meshgrid(x,y)
x_test = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
print("test dataset:", x_test.shape, '\n')

file = open('groundtruth_generation/elliptic_2D_eps_1_8.pkl', 'rb')
u_test = pickle.load(file)
plot_2d(u_test, x, y, path="./elliptic_2D/gt.png")

''' model '''
layers = np.array([2,40,40,40,40,40,40,40,40,1]) #7 hidden layers
PINN = elliptic_2d(layers)
PINN.to(PINN.device)
PINN.bc_weight = 200



def retrain(model, x_train, x_boundary, u_boundary, x_test, u_test, optimizer, epoch, idx=''):
    model.iter = 0
    model.train_model_adam(optimizer, x_boundary, u_boundary, x_train, epoch)

    f = np.abs(model.function(x_test).squeeze().cpu().detach().numpy())
    plot_2d(f.reshape(u_test.shape), x, y, path="./elliptic_2D/res_pinn"+idx+".png")
    print('Test residual MSE:', (np.mean(np.square(f))))

    rmse, u_pred = model.test(x_test,u_test.flatten()[:,None])
    print('Test RMSE:', rmse)
    plot_2d(u_pred.reshape(u_test.shape), x, y, path="./elliptic_2D/pre_pinn"+idx+".png")
    plot_2d(np.abs(u_pred.reshape(u_test.shape)-u_test), x, y, path="./elliptic_2D/diff_pinn"+idx+".png")


N0 = 6000
Nb = 200

x_inner = latin_hypercube_sample([[0,1],[0,1]], N0)
x_l, u_l = boundary_sample([[0,0],[0,1]], Nb)
x_r, u_r = boundary_sample([[1,1],[0,1]], Nb)
x_u, u_u = boundary_sample([[0,1],[0,0]], Nb)
x_b, u_b = boundary_sample([[0,1],[1,1]], Nb)


optimizer = torch.optim.Adam(PINN.parameters(), lr=0.001)
retrain(PINN, np.vstack([x_inner, x_l, x_r, x_u, x_b]), \
    np.vstack([x_l, x_r, x_u, x_b]), \
    np.vstack([u_l, u_r, u_u, u_b]), x_test, u_test, \
    optimizer, 1e5, idx="1")

optimizer = torch.optim.Adam(PINN.parameters(), lr=0.0001)
retrain(PINN, np.vstack([x_inner, x_l, x_r, x_u, x_b]), \
    np.vstack([x_l, x_r, x_u, x_b]), \
    np.vstack([u_l, u_r, u_u, u_b]), x_test, u_test, \
    optimizer, 1e5, idx="2")

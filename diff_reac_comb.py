import numpy as np
import torch
import pickle

from model import diff_reac, latin_hypercube_sample, boundary_sample, plot_2d, plot_1d

''' Settings and seeds '''
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

''' test dataset '''
x = np.linspace(+np.pi/200-np.pi, np.pi-np.pi/200, 200)
y = np.linspace(1/500, 1, 500)
X, Y = np.meshgrid(x,y)
x_test = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
print("test dataset:", x_test.shape, '\n')

file = open('groundtruth_generation/diff_reac_eps_1_10.pkl', 'rb')
u_test = pickle.load(file).T
plot_2d(u_test.T, y, x, path="./diff_reac/gt.png")

''' model '''
layers = np.array([10,40,40,40,40,40,40,40,40,1]) #7 hidden layers
PINN = diff_reac(layers)
PINN.to(PINN.device)
PINN.bc_weight = 200
PINN.fourier = [4,0]


def retrain(model, x_train, x_boundary, u_boundary, x_test, u_test, optimizer, epoch, idx=''):
    model.iter = 0
    model.train_model_adam(optimizer, x_boundary, u_boundary, x_train, epoch)

    f = np.abs(model.function(x_test).squeeze().cpu().detach().numpy())
    plot_2d(f.reshape(u_test.shape).T, y, x, path="./diff_reac/res_comb"+idx+".png")
    print('Test residual MSE:', (np.mean(np.square(f))))

    rmse, u_pred = model.test(x_test,u_test.flatten()[:,None])
    print('Test RMSE:', rmse)
    plot_2d(u_pred.reshape(u_test.shape).T, y, x, path="./diff_reac/pre_comb"+idx+".png")
    plot_2d(np.abs(u_pred.reshape(u_test.shape)-u_test).T, y, x, path="./diff_reac/diff_comb"+idx+".png")
    plot_1d(u_pred.reshape(u_test.shape)[200,:], x, u_gt=u_test[200,:], path="./diff_reac/sec4_comb"+idx+".png")
    plot_1d(u_pred.reshape(u_test.shape)[300,:], x, u_gt=u_test[300,:], path="./diff_reac/sec6_comb"+idx+".png")
    plot_1d(u_pred.reshape(u_test.shape)[400,:], x, u_gt=u_test[400,:], path="./diff_reac/sec8_comb"+idx+".png")



N0 = 1000
Nb = 100
Ni = 100

x_inner = latin_hypercube_sample([[-np.pi,np.pi],[0,.1]], N0)
x_l, u_l = boundary_sample([[-np.pi,np.pi],[0,0]], Ni, gt=lambda x: 0.01*np.sin(x[:,0])[:,None])
x_u, u_u = boundary_sample([[-np.pi,-np.pi],[0,.1]], Nb)
x_b, u_b = boundary_sample([[np.pi,np.pi],[0,.1]], Nb)



optimizer = torch.optim.Adam(PINN.parameters(), lr=0.001)
retrain(PINN, np.vstack([x_inner, x_l, x_u, x_b]), \
    np.vstack([x_l, x_u, x_b]), \
    np.vstack([u_l, u_u, u_b]), x_test, u_test, \
    optimizer, 1e5, idx="1")

optimizer = torch.optim.Adam(PINN.parameters(), lr=0.0001)
retrain(PINN, np.vstack([x_inner, x_l, x_u, x_b]), \
    np.vstack([x_l, x_u, x_b]), \
    np.vstack([u_l, u_u, u_b]), x_test, u_test, \
    optimizer, 1e5, idx="2")

optimizer = torch.optim.Adam(PINN.parameters(), lr=0.00001)
retrain(PINN, np.vstack([x_inner, x_l, x_u, x_b]), \
    np.vstack([x_l, x_u, x_b]), \
    np.vstack([u_l, u_u, u_b]), x_test, u_test, \
    optimizer, 1e5, idx="3")

N0 = 2000
x_inner2 = latin_hypercube_sample([[-np.pi,np.pi],[.1,1]], N0)
x_u, u_u = boundary_sample([[-np.pi,-np.pi],[0,1]], Nb)
x_b, u_b = boundary_sample([[np.pi,np.pi],[0,1]], Nb)



optimizer = torch.optim.Adam(PINN.parameters(), lr=0.001)
retrain(PINN, np.vstack([x_inner, x_inner2, x_l, x_u, x_b]), \
    np.vstack([x_l, x_u, x_b]), \
    np.vstack([u_l, u_u, u_b]), x_test, u_test, \
    optimizer, 1e5, idx="1")

optimizer = torch.optim.Adam(PINN.parameters(), lr=0.0001)
retrain(PINN, np.vstack([x_inner, x_inner2, x_l, x_u, x_b]), \
    np.vstack([x_l, x_u, x_b]), \
    np.vstack([u_l, u_u, u_b]), x_test, u_test, \
    optimizer, 1e5, idx="2")

optimizer = torch.optim.Adam(PINN.parameters(), lr=0.00001)
retrain(PINN, np.vstack([x_inner, x_inner2, x_l, x_u, x_b]), \
    np.vstack([x_l, x_u, x_b]), \
    np.vstack([u_l, u_u, u_b]), x_test, u_test, \
    optimizer, 1e5, idx="3")
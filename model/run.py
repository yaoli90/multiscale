import numpy as np
import torch
import pickle

from visualization import *
from main import *

''' Settings and seeds '''

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device:", device)

if device == 'cuda':
    print(torch.cuda.get_device_name())

''' test dataset '''

x = np.linspace(+np.pi/200-np.pi, np.pi-np.pi/200, 200)
y = np.linspace(1/500, 1, 500)
X, Y = np.meshgrid(x,y)

x_test = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
print("test dataset:", x_test.shape, '\n')

file = open('../groundtruth_generation/diff_reac_eps_1_10.pkl', 'rb')
u_gt = pickle.load(file).T


''' model '''

layers = np.array([10,40,40,40,40,40,40,40,40,1]) #7 hidden layers
PINN = sequential_model(layers, device)
PINN.to(device)
print(PINN)
optimizer = torch.optim.Adam(PINN.parameters(), lr=0.001)
plot_u(u_gt, x, y, title="$u(x)$")


def retrain(model, x_train, x_adv, x_boundary, u_boundary, x_test, u_gt, epoch=2e4, title="0"):

    x_train_temp = np.vstack([x_train, x_adv])

    f_test_pred = np.abs(model.function(x_test).squeeze().cpu().detach().numpy())
    plot_u(f_test_pred.reshape(X.shape), x, y, log=True, title="test $r(x;theta)$")
    print('Test residual MSE: %.5f'  % (np.mean(np.square(f_test_pred))))
    plt.savefig("res"+title+".png")

    model.iter = 0
    model.train_model_adam(optimizer, x_boundary, u_boundary, x_train_temp, epoch)
    rmse, u_pred = model.test(x_test, u_gt.flatten()[:,None])
    
    plot_u(u_pred.reshape(u_gt.shape), x, y, title="predict $u(x)$")
    plt.savefig("u_pred"+title+".png")

    plot_x(u_pred.reshape(u_gt.shape), u_gt, x, pos=[100,200,400])
    plt.savefig("u_pred_line"+title+".png")
    print('Test RMSE: %.5f'  % rmse)
    
    plot_u(np.abs(u_pred.reshape(u_gt.shape)-u_gt), x, y, title="predict $u(x)$")
    plt.savefig("u_diff"+title+".png")

''' k=0 '''
N0 = 4000
x_inner, x_ic, u_ic, x_bc, u_bc = training_data_latin_hypercube([[-np.pi,np.pi],[0,1]], N_inner=N0, N_ic=1000, N_bc=500)
x_train = np.vstack([x_inner, x_ic, x_bc])
x_boundary = np.vstack([x_ic, x_bc])
u_boundary = np.vstack([u_ic, u_bc])


optimizer = torch.optim.Adam(PINN.parameters(), lr=0.001)
retrain(PINN, x_train, np.array([]).reshape((0,2)), x_boundary, u_boundary, x_test, u_gt, epoch=1e5, title="11")
optimizer = torch.optim.Adam(PINN.parameters(), lr=0.0001)
retrain(PINN, x_train, np.array([]).reshape((0,2)), x_boundary, u_boundary, x_test, u_gt, epoch=1e5, title="12")
optimizer = torch.optim.Adam(PINN.parameters(), lr=0.00001)
retrain(PINN, x_train, np.array([]).reshape((0,2)), x_boundary, u_boundary, x_test, u_gt, epoch=1e5, title="13")
plt.close('all')


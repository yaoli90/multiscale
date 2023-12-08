import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
import matplotlib.colors as cls
import numpy as np

def plot_1d(u, x, u_gt=None, ylim=None, path='plot_1d.png'):
    plt.figure(figsize=(4,3))
    plt.plot(x, u)
    if u_gt is not None:
        plt.plot(x, u_gt)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([np.min(x),np.max(x)])
    plt.savefig(path)
    plt.close('all')

def plot_2d(u, x, y, path='plot_2d.png'):
    plt.figure()
    plt.imshow(u, interpolation='nearest', cmap='rainbow',
                extent=[x.min(), x.max(),y.min(), y.max()],
                origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(path)
    plt.close('all')

def plot_u(u, x, t, log=False, title='$u(x;theta)$'):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-0.46, left=0.15, right=0.55, wspace=0)
    ax = plt.subplot(gs0[:, :])
    if not log:
        h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    else:
        h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    #ax.set_aspect('equal')
    ax.set_xlabel('$y$')
    ax.set_ylabel('$x$')
    ax.set_title(title, fontsize = 10)
    return ax

def plot_x(u, U_gt, x, pos):
    fig, ax = plt.subplots()
    gs1 = gridspec.GridSpec(1, 3)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,U_gt[pos[0],:], 'b-', linewidth = 1, label = 'Exact')
    ax.plot(x,u[pos[0],:], 'r--', linewidth = 1, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0.25s$', fontsize = 10)
    ax.set_ylim([-.025,.025])



    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,U_gt[pos[1],:], 'b-', linewidth = 1, label = 'Exact')
    ax.plot(x,u[pos[1],:], 'r--', linewidth = 1, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0.50s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    ax.set_ylim([-.025,.025])

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,U_gt[pos[2],:], 'b-', linewidth = 1, label = 'Exact')
    ax.plot(x,u[pos[2],:], 'r--', linewidth = 1, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0.75s$', fontsize = 10)
    ax.set_ylim([-.025,.025])

def plot_u_x(u, U_gt, x, y, pos=192, title='$u(x;theta)$'):
    ax = plot_u(u, x, y, title=title)
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(y[pos]*np.ones((2,1)), line, 'w-', linewidth = 1)
    plot_x(u, U_gt, x, pos)
    plt.show()

def plot_samples(s, f):



    sample_mse = np.mean(np.square(f))
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-0.46, left=0.15, right=0.55, wspace=0)
    ax = plt.subplot(gs0[:, :])
    h = ax.scatter(s[:,1],s[:,0], c=f, cmap="rainbow", norm=cls.LogNorm(vmin=1e-1), s=15)
    ax.set_xlabel('adversarial samples, %.5f' % sample_mse)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.axis('square')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_xlabel('$y$')
    ax.set_ylabel('$x$')
    fig.colorbar(h, cax=cax)
    plt.show()

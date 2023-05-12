import torch
import numpy as np
import json
from data.data import get_iter
from scipy.stats import norm

def curve(model, test_matrix, t_grid, targetreg=None):
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg is None:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            out = out[1].data.squeeze()
            out = out.mean()
            t_grid_hat[1, _] = out
            mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse
    else:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
            out = out.mean()
            t_grid_hat[1, _] = out
            mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse


# type = "TR" or "DR"
def pseudo_outcome(model, test_matrix, t_grid, type = 'DR'):
    n_test = t_grid.shape[1]
    PO = torch.zeros(n_test)

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
    if type == 'DR':
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            g = out[0].data.squeeze()
            out = out[1].data.squeeze()
            PO[_] = (t_grid[1,_] - out[_]) * g.mean() / (g[_] + 1e-6) + out.mean()
    else if type == 'TR':
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze()
            PO[_] = out[_] + tr_out/ (g[_] + 1e-6)
    return PO

def test(PO, t_grid_hat, rho = [0.01, 0.05, 0.1, 0.2, 0.5]):
    rho = np.array(rho)
    n_test = t_grid_hat.shape[1]
    delta = ((PO - t_grid_hat[1, :]) ** 2 - (PO - t_grid_hat[1, :].mean()) ** 2) + rho.reshape(-1,1) * np.random.normal(size = n_test)
    theta = delta.sum(axis = 1) / (np.sqrt(n_test) * delta.std(axis = 1)) # test statistic
    p_val = norm.cdf(theta)
    return p_val

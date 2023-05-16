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

def test(model, test_matrix, t_grid_hat, rho = [0.01, 0.05, 0.1, 0.2, 0.3]):
    n_test = test_matrix.shape[0]
    mu = torch.zeros(n_test, n_test)
    pi = mu

    test_loader = get_iter(test_matrix, batch_size=n_test, shuffle=False)
    for _ in range(n_test):
        for idx, (inputs, y) in enumerate(test_loader):
            t = inputs[:, 0]
            t *= 0
            t += inputs[_, 0]
            x = inputs[:, 1:]
            break
        out = model.forward(t, x)
        mu[_,:] = out[0].data.squeeze()
        pi[_,:] = out[1].data.squeeze()

    mu_mean = torch.mean(mu, 1, True)
    pi_mean = torch.mean(pi, 1, True)
    y_hat = (y.repeat(n_test, 1) - mu) / pi * pi_mean.repeat(1, n_test) + mu_mean.repeat(1, n_test)
    g_hat = t_grid_hat[1]
    g_tilde = torch.mean(g_hat).repeat(n_test)
    delta = torch.mean((y_hat - torch.reshape(g_hat, (n_test,1)).repeat(1, n_test)) ** 2, 1) - torch.mean((y_hat - torch.reshape(g_tilde, (n_test,1)).repeat(1, n_test)) ** 2, 1)

    rho
    rho = np.array(rho)
    delta = delta.repeat(len(rho), 1).numpy()
    noise = rho.reshape(-1,1) * np.random.normal(size = n_test)
    theta = delta.sum(axis = 1) / (np.sqrt(n_test) * delta.std(axis = 1))
    p_val = norm.cdf(theta)
    return p_val

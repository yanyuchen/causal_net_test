import torch
import numpy as np
import json
from data.data import get_iter
from scipy.stats import norm
import os
import math

def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()

def curve(model, test_matrix, t_grid, targetreg, arange = [0.01, 0.99], pi_low = 1e-6):
    mask = np.logical_or(test_matrix[:,0] > arange[0], test_matrix[:,0] < arange[1]).bool()
    test_matrix = test_matrix[mask,:]
    t_grid = t_grid[:,mask]

    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

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
        out = out[1].data.squeeze() + tr_out / (g + pi_low)
        out = out.mean()
        t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
    return t_grid_hat, mse

def calculate_delta(model, test_matrix, t_grid_hat, targetreg, arange = [0.01, 0.99], pi_low = 1e-6):
    mask = np.logical_or(test_matrix[:,0] > arange[0], test_matrix[:,0] < arange[1]).bool()
    test_matrix = test_matrix[mask,:]

    n_test = test_matrix.shape[0]
    mu_tr = torch.zeros(n_test, n_test)

    test_loader = get_iter(test_matrix, batch_size=n_test, shuffle=False)
    for _ in range(n_test):
        for idx, (inputs, y) in enumerate(test_loader):
            t = inputs[:, 0]
            t *= 0
            t += t_grid_hat[0, _]
            x = inputs[:, 1:]
            break
        out = model.forward(t, x)
        g = out[0].data.squeeze()
        out = out[1].data.squeeze()

        tr_out = targetreg(t).data
        mu_tr[_,:] = out + tr_out / (g + pi_low)

    g_hat = t_grid_hat[1]
    g_tilde = torch.mean(g_hat).repeat(n_test)
    delta = torch.mean((mu_tr - torch.reshape(g_hat, (n_test,1)).repeat(1, n_test)) ** 2, 1) - torch.mean((mu_tr - torch.reshape(g_tilde, (n_test,1)).repeat(1, n_test)) ** 2, 1)
    return delta.numpy()

def test_from_delta(delta, rho):
    n_test = delta.shape
    delta += rho * np.random.normal(size = n_test)
    theta = np.sqrt(n_test) * delta.mean() / delta.std()
    p_val = norm.cdf(theta)
    return p_val

def test_given_ratio(model, test_matrix, t_grid_hat, rho, targetreg, arange = [0.01, 0.99], pi_low = 1e-6):
    delta = calculate_delta(model, test_matrix, t_grid_hat, targetreg, arange, pi_low)
    p_val = test_from_delta(delta, rho)
    return p_val

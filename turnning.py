import torch
import math
import numpy as np
import os
import pandas as pd

from models.dynamic_net import Vcnet, TR #Drnet
from data.data import get_iter, split
from utils.eval import *

import argparse

def calculate_delta(model, test_matrix, t_grid_hat, targetreg):
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
        mu_tr[_,:] = out + tr_out / (g + 1e-6)

    g_hat = t_grid_hat[1]
    g_tilde = torch.mean(g_hat).repeat(n_test)
    delta = torch.mean((mu_tr - torch.reshape(g_hat, (n_test,1)).repeat(1, n_test)) ** 2, 1) - torch.mean((mu_tr - torch.reshape(g_tilde, (n_test,1)).repeat(1, n_test)) ** 2, 1)
    return delta.numpy()

def calculate_U_delta(args, delta, alpha = 0.05, U = 5, inf_ratio = 0.15):
    load_path = args.data_dir
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = pd.read_csv(load_path + f'/delta_{delta}_data.txt', header=None, sep=' ')
    data = data.to_numpy()
    t_grid_dat = pd.read_csv(load_path + f'/delta_{delta}_t_grid.txt', header=None, sep=' ')
    t_grid_dat = t_grid_dat.to_numpy()

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9
    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # choose from {'Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr'}
    method_list = ['Vcnet_tr']

    # delta
    n_sample = data.shape[0]
    n_train = round(n_sample * (1-inf_ratio)) + 1
    n_test = n_sample - n_train
    delta = np.zeros(n_test * U).reshape(U, n_test)
    for _ in range(U):
        perm = np.random.permutation(data.shape[0])
        data[:,0] = data[perm, 0]
        t_grid_dat[0, :] = t_grid_dat[0, perm]
        train_matrix, test_matrix, t_grid = split(data, t_grid_dat, inf_ratio)

        train_loader = get_iter(train_matrix, batch_size=train_matrix.shape[0], shuffle=True)
        test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

        for model_name in method_list:
            # import model
            if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
                cfg_density = [(4, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                num_grid = 10
                cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                degree = 2
                knots = [0.33, 0.66]
                model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
                model._initialize_weights()

            elif model_name == 'Drnet' or model_name == 'Drnet_tr':
                cfg_density = [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                num_grid = 10
                cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                isenhance = 1
                model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
                model._initialize_weights()

            elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
                cfg_density = [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                num_grid = 10
                cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                isenhance = 0
                model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
                model._initialize_weights()

            # use Target Regularization?
            if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Tarnet_tr':
                isTargetReg = 1
            else:
                isTargetReg = 0

            if isTargetReg:
                tr_knots = list(np.arange(0.1, 1, 0.1))
                tr_degree = 2
                TargetReg = TR(tr_degree, tr_knots)
                TargetReg._initialize_weights()

            # best cfg for each model
            if model_name == 'Tarnet':
                init_lr = 0.05
                alpha = 1.0
            elif model_name == 'Tarnet_tr':
                init_lr = 0.05
                alpha = 0.5
                tr_init_lr = 0.001
                beta = 1.
            elif model_name == 'Drnet':
                init_lr = 0.05
                alpha = 1.
            elif model_name == 'Drnet_tr':
                init_lr = 0.05
                # init_lr = 0.05 tuned
                alpha = 0.5
                tr_init_lr = 0.001
                beta = 1.
            elif model_name == 'Vcnet':
                init_lr = 0.0001
                alpha = 0.5
            elif model_name == 'Vcnet_tr':
                init_lr = 0.0001
                alpha = 0.5
                tr_init_lr = 0.001
                beta = 1.

            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

            if isTargetReg:
                tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

            print('model = ', model_name)
            loss_values = []
            for epoch in range(num_epoch):
                for idx, (inputs, y) in enumerate(train_loader):
                    t = inputs[:, 0]
                    x = inputs[:, 1:]

                    if isTargetReg:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        trg = TargetReg(t)
                        loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                        loss.backward()
                        optimizer.step()

                        tr_optimizer.zero_grad()
                        out = model.forward(t, x)
                        trg = TargetReg(t)
                        tr_loss = criterion_TR(out, trg, y, beta=beta)
                        tr_loss.backward()
                        tr_optimizer.step()
                    else:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        loss = criterion(out, y, alpha=alpha)
                        loss.backward()
                        optimizer.step()

                    loss_values.append(loss.data)

                if epoch % verbose == 0:
                    print(f'iteration: {_ + 1}/{U}')
                    print('current epoch: ', epoch)
                    print('loss: ', loss.data)

            if isTargetReg:
                t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)
            else:
                t_grid_hat, mse = curve(model, test_matrix, t_grid)

            mse = float(mse)

            delta[_, :] = calculate_delta(model, test_matrix, t_grid_hat, TargetReg)
    return delta

def turn_rho(delta0, rho, alpha = 0.05):
    U, n_test = delta0.shape
    p_val = np.zeros(len(rho) * U).reshape(len(rho), U)
    rej = np.zeros(len(rho))
    for _ in range(len(rho)):
        noise = rho[_]
        delta = delta0 + noise * np.random.normal(size = U * n_test).reshape(U, n_test)
        theta = delta.sum(axis = 1) / (np.sqrt(n_test) * delta.std(axis = 1))
        p_val[_,:] = norm.cdf(theta)
        rej[_] = (p_val[_,:] <= alpha).mean()
    return p_val, rej

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/simu2/eval/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu2/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=1000, help='num of epochs to train') # 800 # 80000 #260000

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    args = parser.parse_args()

    for inf_ratio in [0.1, 0.15, 0.3, 0.4]:
        for delta in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f'delta: {delta}')
            print(f'inf_ratio: {inf_ratio}')
            Delta = calculate_U_delta(args, delta, alpha = 0.05, U = 100, inf_ratio = inf_ratio)
            np.savetxt(args.save_dir + f'/Delta_delta_{delta}_inf_ratio_{inf_ratio}.txt', Delta)

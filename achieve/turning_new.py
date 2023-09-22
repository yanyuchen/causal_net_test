import torch
import numpy as np
import pandas as pd

import os
import math
import warnings
import argparse

from models.dynamic_net import Vcnet, TR #Drnet
from data.data import get_iter, split
from utils.eval import *

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/simu2/eval/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu2/tune', help='dir to save result')

    # common
    #parser.add_argument('--num_dataset', type=int, default=2, help='num of datasets to train')
    parser.add_argument('--U', type=int, default=100, help='num of permutation')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train') # 800 # 80000 #260000

    # print train info
    parser.add_argument('--verbose', type=bool, default=False, help='print train info freq or not')
    parser.add_argument('--verbose_num', type=int, default=200, help='number of epochs to print train info')

    args = parser.parse_args()

    #inf_ratio_candidate = [0.08, 0.1]
    inf_ratio_candidate = [0.2]

    # six scenario
    #delta_list = [x/10 for x in range(0, 6, 1)]
    #delta_list = [0, 0.5, 1] #[x/10 for x in range(0, 6, 1)] #[0, 0.5]
    delta_list = [0, 0.3, 0.5]

    # data
    load_path = args.data_dir

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # fixed parameter for optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose_num

    cfg_density = [(4, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    #cfg_density = [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    num_grid = 10
    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
    degree = 2
    knots = [0.33, 0.66]
    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)

    tr_knots = list(np.arange(0.1, 1, 0.1))
    tr_degree = 2
    TargetReg = TR(tr_degree, tr_knots)

    init_lr = 0.0001
    alpha = 0.5
    tr_init_lr = 0.001
    beta = 1.

    for inf_ratio in inf_ratio_candidate:
        for delta in delta_list:
            print(f'Start the case for delta = {delta}; inf_ratio = {inf_ratio}')

            data = pd.read_csv(load_path + f'/delta_{delta}_data.txt', header=None, sep=' ')
            data = data.to_numpy()
            t_grid_dat = pd.read_csv(load_path + f'/delta_{delta}_t_grid.txt', header=None, sep=' ')
            t_grid_dat = t_grid_dat.to_numpy()

            # Delta statistic
            n_sample = data.shape[0]
            n_train = round(n_sample * (1-inf_ratio)) + 1
            n_test = n_sample - n_train
            Delta = np.zeros(401 * args.U).reshape(args.U, 401)

            for _ in range(args.U):
                print(f'permutation: {_ + 1}/{args.U}')
                perm = np.random.permutation(data.shape[0])
                data[:,0] = data[perm, 0]
                t_grid_dat[0, :] = t_grid_dat[0, perm]
                train_matrix, test_matrix, t_grid = split(data, t_grid_dat, inf_ratio)

                train_loader = get_iter(train_matrix, batch_size=train_matrix.shape[0], shuffle=True)
                test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

                # reinitialize model
                model._initialize_weights()
                TargetReg._initialize_weights()

                # define optimizer
                optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)
                tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

                for epoch in range(num_epoch):
                    for idx, (inputs, y) in enumerate(train_loader):
                        t = inputs[:, 0]
                        x = inputs[:, 1:]

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

                    if args.verbose == True:
                        if epoch % verbose == 0:
                            print('current epoch: ', epoch)
                            print('loss: ', loss.data)

                #t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)

                #Delta[_, :] = calculate_delta(model, test_matrix, t_grid_hat, TargetReg)
                Delta[_,:] = calculate_delta0(model, test_matrix, targetreg=TargetReg)
            np.savetxt(args.save_dir + f'/Delta_delta_{delta}_inf_ratio_{inf_ratio}_U_{args.U}.txt', Delta)
            print('Delta statistics saved to', save_path)
            print(f'End the case for delta = {delta}; inf_ratio = {inf_ratio}')
            print('-----------------------------------------------------------------')
            print('-----------------------------------------------------------------')

    print('Program done. Complete all Delta statistics')

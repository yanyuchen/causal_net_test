import torch
import math
import numpy as np
import os
import pandas as pd

from models.dynamic_net import Vcnet, TR #Drnet
from data.data import get_iter, split
from utils.eval import *

import argparse

import matplotlib.pyplot as plt

def save_checkpoint(state, model_name='', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/simu2/eval/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu2/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=1000, help='num of epochs to train') # 800 # 80000 #260000

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')

    args = parser.parse_args()

    # delta
    delta = 0

    # splitting ratio, inf_ratio; noise size, rho
    inf_ratio = 0.3
    rho = 0.4

    # dir
    load_path = args.data_dir
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = pd.read_csv(load_path + f'/delta_{delta}_data.txt', header=None, sep=' ')
    data = data.to_numpy()
    t_grid_dat = pd.read_csv(load_path + f'/delta_{delta}_t_grid.txt', header=None, sep=' ')
    t_grid_dat = t_grid_dat.to_numpy()
    train_matrix, test_matrix, t_grid = split(data, t_grid_dat, inf_ratio)

    train_loader = get_iter(train_matrix, batch_size=train_matrix.shape[0], shuffle=True)
    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9
    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    grid = []
    MSE = []

    # choose from {'Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr'}
    method_list = ['Vcnet_tr']

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
                print('current epoch: ', epoch)
                print('loss: ', loss.data)



        if isTargetReg:
            t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)
        else:
            t_grid_hat, mse = curve(model, test_matrix, t_grid)

        mse = float(mse)
        print('current loss: ', float(loss.data))
        print('current test loss: ', mse)
        print('-----------------------------------------------------------------')

        save_checkpoint({
            'model': model_name,
            'best_test_loss': mse,
            'model_state_dict': model.state_dict(),
            'TR_state_dict': TargetReg.state_dict() if isTargetReg else None,
        }, model_name=model_name, checkpoint_dir=save_path)

        print('-----------------------------------------------------------------')

        grid.append(t_grid_hat)
        MSE.append(mse)

        plt.plot(loss_values, label = 'train')
        plt.legend(loc='upper right')
        #plt.show()
        plt.savefig(save_path + "/train_loss.pdf", bbox_inches='tight')

        p_val = test_given_ratio(model, test_matrix, t_grid_hat, rho, TargetReg)
        print('p_value: ', p_val)



    if args.plt_adrf:
        import matplotlib.pyplot as plt

        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 22,
        }

        font_legend = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 22,
        }
        plt.figure(figsize=(5, 5))

        c1 = 'gold'
        c2 = 'red'
        #c3 = 'dodgerblue'

        truth_grid = t_grid[:,t_grid[0,:].argsort()]
        x = truth_grid[0, :]
        y = truth_grid[1, :]
        plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)

        x = grid[0][0, :]
        y = grid[0][1, :]
        plt.scatter(x, y, marker='h', label='Vcnet', alpha=1, zorder=2, color=c2, s=20)

        #x = grid[0][0, :]
        #y = grid[0][1, :]
        #plt.scatter(x, y, marker='H', label='Drnet', alpha=1, zorder=3, color=c3, s=20)

        plt.yticks(np.arange(-2.0, 1.1, 0.5), fontsize=0, family='Times New Roman')
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=0, family='Times New Roman')
        plt.grid()
        plt.legend(prop=font_legend, loc='lower left')
        plt.xlabel('Treatment', font1)
        plt.ylabel('Response', font1)

        plt.savefig(save_path + "/Vc_Dr.pdf", bbox_inches='tight')

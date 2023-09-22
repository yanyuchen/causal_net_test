import torch
import numpy as np
import pandas as pd

import math
import time
import os
import argparse

from data.simu4 import *
from data.data import *
from utils.eval import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/simu4/eval', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu4/eval', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=200, help='num of datasets to train')

    # significance level
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level for test')

    args = parser.parse_args()

    # six scenario
    #delta_list = [x/10 for x in range(0, 6, 1)]
    delta_list = [0, 0.3, 0.5] #[x/10 for x in range(0, 6, 1)] #[0, 0.5]

    # splitting ratio, inf_ratio; noise size, rho
    inf_ratio = 0.2 #0.1 #0.15 #0.3
    rho = 0.11 #0.135 #0.15 #0.4

    size = 400
    a = 0.01
    b = 0.99
    step = (b - a)/ size

    arange = np.arange(a, b + step, step)
    arange = torch.from_numpy(arange)
    repeating_pattern = np.tile(np.array([4, 2]), 199)
    repeating_pattern = np.concatenate((np.array([1]), repeating_pattern, np.array([4, 1]))) / 3 * step


    # data
    load_path = args.data_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for delta in delta_list:
        print(f'Start the case for delta = {delta}')

        p_val = np.zeros(num_dataset)
        run_time = np.zeros(num_dataset)
        for _ in range(num_dataset):
            print(f'dataset: {_ + 1}/{num_dataset}')

            data = pd.read_csv(load_path + '/' + str(_) + f'/delta_{delta}_data.txt', header=None, sep=' ')
            data = data.to_numpy()
            t_grid_dat = pd.read_csv(load_path + '/' + str(_) + f'/delta_{delta}_t_grid.txt', header=None, sep=' ')
            t_grid_dat = t_grid_dat.to_numpy()

            each_fold = kfold(data, t_grid_dat, inf_ratio)
            k = len(each_fold)
            Delta_all = []

            # get the start time
            st = time.time()
            for i in range(k):
                print(f'ford: {i + 1}/{k}')
                train_matrix, test_matrix, t_grid = each_fold[i]

                n_test = test_matrix.shape[0]
                g_hat0 = torch.zeros(1, n_test)
                mu_tr = torch.zeros(arange.shape[0], n_test)

                for j in range(arange.shape[0]):
                    t = arange[j]
                    for i in range(n_test):
                        x = test_matrix[i,1:5]
                        mu_tr[j,i] = t_x_y(t, x, delta)
                g_hat = mu_tr.mean(1)

                for i in range(n_test):
                    t = test_matrix[i,0]
                    x = test_matrix[i,1:5]
                    out = t_x_y(t, x, delta)
                    g_hat0[0, i] = out.mean()
                g_tilde = torch.mean(g_hat).repeat(arange.shape[0])

                la = repeating_pattern @ (mu_tr - torch.reshape(g_hat, (arange.shape[0],1)).repeat(1, n_test) ** 2).numpy()
                l0 = repeating_pattern @ (mu_tr - torch.reshape(g_tilde, (arange.shape[0],1)).repeat(1, n_test) ** 2).numpy()
                Delta = la - l0
                Delta_all += Delta.tolist()

            p_val0 = test_from_delta(np.array(Delta_all), rho)
            # get the end time
            et = time.time()
            # get the execution time
            elapsed_time = et - st

            run_time[_] = elapsed_time
            #print('p_value: ', p_val0)
            p_val[_] = p_val0

        print(f'End the case for delta = {delta}')
        data_file = os.path.join(save_path, f'p_val_oracle_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt')
        np.savetxt(data_file, p_val)
        print('p-values saved to', save_path)
        time_file = os.path.join(save_path, f'run_time_oracle_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt')
        np.savetxt(time_file, run_time)
        print('run time saved to', save_path)
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')

    print('Complete all p-values; now calculate the rejection rate...')
    rej_rate = np.zeros(len(delta_list))
    time_cost = np.zeros(len(delta_list))
    for _ in range(len(delta_list)):
        delta = delta_list[_]
        try:
            p_val = pd.read_csv(save_path + f'/p_val_oracle_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt', header = None)
            run_time = pd.read_csv(save_path + f'/run_time_oracle_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt', header = None)
        except FileNotFoundError:
            continue
        p_val = p_val.to_numpy()
        rej_rate[_] = (p_val < args.alpha).mean()

        run_time = run_time.to_numpy()
        time_cost[_] = run_time.mean()

    data_file = os.path.join(save_path, f'rej_rate_oracle_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt')
    np.savetxt(data_file, np.array([delta_list, rej_rate, time_cost]))
    print('rejection rate done, saved to', save_path)
    print('Program done')

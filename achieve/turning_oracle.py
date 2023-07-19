import torch
import numpy as np

import os
import math
import argparse

import data.simu2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/simu2/tune/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu2/tune', help='dir to save result')

    # common
    #parser.add_argument('--num_dataset', type=int, default=2, help='num of datasets to train')
    parser.add_argument('--U', type=int, default=100, help='num of permutation')

    args = parser.parse_args()

    inf_ratio_candidate = [0.08, 0.1]

    # six scenario
    #delta_list = [x/10 for x in range(0, 6, 1)]
    delta_list = [0, 0.5] #[x/10 for x in range(0, 6, 1)] #[0, 0.5]

    # data
    load_path = args.data_dir

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
            Delta = np.zeros(n_test * args.U).reshape(args.U, n_test)

            for _ in range(args.U):
                print(f'permutation: {_ + 1}/{args.U}')
                perm = np.random.permutation(data.shape[0])
                data[:,0] = data[perm, 0]
                t_grid_dat[0, :] = t_grid_dat[0, perm]
                train_matrix, test_matrix, t_grid = split(data, t_grid_dat, inf_ratio)

                g_hat = t_grid[1]
                g_tilde = torch.mean(g_hat).repeat(n_test)

            print(f'End the case for delta = {delta}; inf_ratio = {inf_ratio}')

            np.savetxt(args.save_dir + f'/oracle_Delta_delta_{delta}_inf_ratio_{inf_ratio}_U_{args.U}.txt', Delta)
            print('Delta statistics saved to', save_path)
            print('-----------------------------------------------------------------')
            print('-----------------------------------------------------------------')

    print('Program done. Complete all Delta statistics')

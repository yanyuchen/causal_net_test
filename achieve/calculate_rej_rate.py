import numpy as np
import pandas as pd
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--save_dir', type=str, default='logs/simu2/eval', help='dir to save result')

    args = parser.parse_args()

    # delta
    delta_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # splitting ratio, inf_ratio; noise size, rho
    inf_ratio = 0.3
    rho = 0.4

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # significance level
    alpha = 0.05

    rej_rate = np.zeros(len(delta_list))
    for _ in range(len(delta_list)):
        delta = delta_list[_]
        try:
            p_val = pd.read_csv(save_path + f'/p_val_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}.txt', header = None)
        except FileNotFoundError:
            continue
        p_val = p_val.to_numpy()
        rej_rate[_] = (p_val < alpha).mean()

    data_file = os.path.join(save_path, f'rej_rate_at_inf_ratio_{inf_ratio}_rho_{rho}.txt')
    np.savetxt(data_file, np.array([delta_list, rej_rate]))

import os
import numpy as np

from data.simu2 import simu_data2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate simulate data')
    parser.add_argument('--save_dir', type=str, default='dataset/simu2', help='dir to save generated data')
    parser.add_argument('--num_eval', type=int, default=1000, help='num of dataset for evaluating the methods') #100
    parser.add_argument('--num_tune', type=int, default=0, help='num of dataset for tuning the parameters') #20
    parser.add_argument('--num_obs', type=int, default=1000, help='num of observation in one dataset')

    args = parser.parse_args()
    save_path = args.save_dir

    for delta in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for _ in range(args.num_tune):
            print('delta: ', delta)
            print('generating tuning set: ', _)
            data_path = os.path.join(save_path, 'tune', str(_))
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            train_matrix, t_grid = simu_data2(args.num_obs, delta) #500, 200

            data_file = os.path.join(data_path, f'delta_{delta}_data.txt')
            np.savetxt(data_file, train_matrix.numpy())

            data_file = os.path.join(data_path, f'delta_{delta}_t_grid.txt')
            np.savetxt(data_file, t_grid.numpy())


        for _ in range(args.num_eval):
            print('delta: ', delta)
            print('generating eval set: ', _)
            data_path = os.path.join(save_path, 'eval', str(_))
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            train_matrix, t_grid = simu_data2(args.num_obs, delta)

            data_file = os.path.join(data_path, f'delta_{delta}_data.txt')
            np.savetxt(data_file, train_matrix.numpy())

            data_file = os.path.join(data_path, f'delta_{delta}_t_grid.txt')
            np.savetxt(data_file, t_grid.numpy())

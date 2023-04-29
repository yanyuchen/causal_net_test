import os
import numpy as np

from data.simu2 import simu_data2
import argparse

if __name__ == "__main__":
    delta = 0.5

    parser = argparse.ArgumentParser(description='generate simulate data')
    parser.add_argument('--save_dir', type=str, default='dataset/simu2', help='dir to save generated data')
    parser.add_argument('--num_eval', type=int, default=1, help='num of dataset for evaluating the methods') #100
    parser.add_argument('--num_tune', type=int, default=1, help='num of dataset for tuning the parameters') #20

    args = parser.parse_args()
    save_path = args.save_dir

    for _ in range(args.num_tune):
        print('generating tuning set: ', _)
        data_path = os.path.join(save_path, 'tune', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_matrix, test_matrix, t_grid = simu_data2(500, 200, delta) #500, 200

        data_file = os.path.join(data_path, 'train.txt')
        np.savetxt(data_file, train_matrix.numpy())
        data_file = os.path.join(data_path, 'test.txt')
        np.savetxt(data_file, test_matrix.numpy())
        data_file = os.path.join(data_path, 't_grid.txt')
        np.savetxt(data_file, t_grid.numpy())

    for _ in range(args.num_eval):
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'eval', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_matrix, test_matrix, t_grid = simu_data2(500, 200, delta) #500, 200

        data_file = os.path.join(data_path, 'train.txt')
        np.savetxt(data_file, train_matrix.numpy())
        data_file = os.path.join(data_path, 'test.txt')
        np.savetxt(data_file, test_matrix.numpy())
        data_file = os.path.join(data_path, 't_grid.txt')
        np.savetxt(data_file, t_grid.numpy())

import torch
from torch.distributions import Beta

def x_lam(x):
    c1 = torch.tensor([0.1, 0.1, -0.1, 0.2])
    lam = torch.sigmoid(torch.dot(x, c1))
    return lam

def x_t(x):
  lam = x_lam(x)
  t = Beta(lam, 1-lam)
  return t.sample()

def t_x_y(t, x, delta):
  gt = delta * torch.exp(-100 * (t-0.5) ** 2)
  y = torch.dot(x, torch.tensor([0.2, 0.2, 0.3, -0.1])) + t * torch.dot(x, torch.tensor([-0.1, 0, 0.1, 0])) + gt
  return y

def simu_data2(n_train, n_test, delta):
    train_matrix = torch.zeros(n_train, 6)
    test_matrix = torch.zeros(n_test, 6)
    for _ in range(n_train):
        x = torch.rand(4)
        train_matrix[_, 1:5] = x
        t = x_t(x)
        train_matrix[_, 0] = t

        y = t_x_y(t, x, delta)
        y += torch.randn(1)[0] * 0.5

        train_matrix[_, -1] = y

    for _ in range(n_test):
        x = torch.rand(4)
        test_matrix[_, 1:5] = x
        t = x_t(x)
        test_matrix[_, 0] = t

        y = t_x_y(t, x, delta)
        y += torch.randn(1)[0] * 0.5

        test_matrix[_, -1] = y

    t_grid = torch.zeros(2, n_test)
    t_grid[0, :] = test_matrix[:, 0].squeeze()

    for i in range(n_test):
        psi = 0
        t = t_grid[0, i]
        for j in range(n_test):
            x = test_matrix[j, 1:5]
            psi += t_x_y(t, x, delta)
        psi /= n_test
        t_grid[1, i] = psi

    return train_matrix, test_matrix, t_grid

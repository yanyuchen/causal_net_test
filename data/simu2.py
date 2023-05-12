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

def g_t(t, delta):
    return delta * torch.exp(-100 * (t-0.5) ** 2)

def t_x_y(t, x, delta):
    gt = g_t(t, delta)
    y = torch.dot(x, torch.tensor([0.2, 0.2, 0.3, -0.1])) + t * torch.dot(x, torch.tensor([-0.1, 0, 0.1, 0])) + gt
    return y

def simu_data2(n_train, delta):
    train_matrix = torch.zeros(n_train, 6)

    for _ in range(n_train):
        x = torch.rand(4)
        train_matrix[_, 1:5] = x
        t = x_t(x)
        train_matrix[_, 0] = t

        y = t_x_y(t, x, delta)
        y += torch.randn(1)[0] * 0.5

        train_matrix[_, -1] = y

    t_grid = torch.zeros(2, n_train)
    t_grid[0, :] = train_matrix[:, 0].squeeze()
    t_gird[1, :] = g_t(t_grid[0, :], delta)

    return train_matrix, t_grid

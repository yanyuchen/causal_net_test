from scipy.stats import norm, uniform
import statsmodels.api as sm
import pylab
import pandas as pd
import numpy as np

import argparse

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
    delta = 0
    inf_ratio = 0.4
    Delta = pd.read_csv(args.save_dir + f'/Delta_delta_{delta}_inf_ratio_{inf_ratio}.txt', header=None, sep=' ')
    rho = [0.2, 0.3, 0.4, 0.5, 1, 2, 2.5] #0.4

    rej = np.zeros(len(rho))
    num = 1000
    for i in range(num):
        p_val, rej0 = turn_rho(Delta , rho)
        rej += rej0
    rej /= num
    rej

    sm.qqplot(p_val[2,:], uniform, line = '45')
    pylab.show()

# coding: utf-8

import numpy as np
import torch
import matplotlib.pyplot as plt
import math 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rbf_covariance(x, xp, var, lengthscale):
    diff = (x / lengthscale).unsqueeze(1) - (xp / lengthscale).unsqueeze(0)
    return var * torch.exp(-0.5 * torch.sum(diff.pow(2), 2))


def get_predictive_functions(X, X_star, y, var=1.0,
        lengthscale=1.0, noise=0.1):
    K = rbf_covariance(X, X, var, lengthscale)
    K_star = rbf_covariance(X, X_star, var, lengthscale)
    K_dstar = rbf_covariance(X_star, X_star, var, lengthscale)
    
    L = torch.potrf(K + noise * torch.eye(len(y)))
    L_inv = torch.inverse(L)
    alpha = torch.mm(torch.inverse(L.t()), torch.mm(L_inv, y))
    pred_mu = torch.mm(K_star.t(), alpha)
    pred_mu = pred_mu.view((-1, 1))

    v = torch.mm(L_inv, K_star)
    # pred_cov = K_dstar - torch.mm(v.t(), v)
    pred_cov = K_dstar - K_star.t().mm(torch.inverse(
                K + noise * torch.eye(len(y)))).mm(K_star)
    log_ml = -0.5 * torch.mm(y.t(), alpha) - torch.sum(torch.diag(L)) \
                - 0.5 * len(y) * torch.log(torch.Tensor([math.pi]))
    
    return pred_mu, pred_cov, log_ml


def build_toy_datasets(D=1, n_data=20, noise_std=0.1):
    
    inputs = torch.cat([torch.linspace(0, 3, int(n_data/2)),
                torch.linspace(6, 8, int(n_data/2))])
    targets = (torch.cos(inputs) + torch.randn(n_data) * noise_std) / 2.0
    targets = targets.view((len(targets), 1))
    inputs = (inputs - 4.0) / 2.0 
    inputs = inputs.view((len(inputs), D))

    return inputs, targets 

if __name__ == "__main__":
    D = 1
    var = 1.0
    lengthscale = 1.0
    noise = 0.1
    X, y = build_toy_datasets(D=D)
    X_star = torch.linspace(-7, 7, 300).view((300, 1))

    mu, cov, log_ml = get_predictive_functions(X, X_star, y)
    print('Log Marginal Likelihood:', log_ml.item())

    std = torch.sqrt(torch.diag(cov))

    mu = mu.numpy().reshape(-1)
    cov = cov.numpy()
    std = std.numpy()
    X_star = X_star.numpy()
    X = X.numpy()
    y = y.numpy()

    plt.plot(X_star, mu, 'b')
    plt.fill(np.concatenate([X_star, X_star[::-1]]),
            np.concatenate([mu - 1.96*std,
            (mu + 1.96*std)[::-1]]),
            alpha=.15, fc='Blue', ec="None")
    plt.plot(X, y, 'kx')
    sampled_func = np.random.multivariate_normal(mu, cov, size=5)
    plt.plot(X_star, sampled_func.T)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig("gp.png", dpi=300)

import argparse

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(dest='input_files', nargs='*',
                    help='''Generate as many samples you want with e.g.
                            (replace N with appropriate number)
                            ```
                                python3 -m simulate -n 10000 -i N
                            ```
                            Analyse with `python -m diagnostic sample-10000-*.npz`.
                            ''')
args = parser.parse_args()

samples = []
for file_name in args.input_files:
    data = np.load(file_name)['data']
    samples += [data]
sample = np.concatenate(samples, axis=0)

print(f'Sample shape      : {sample.shape}')

n_theta = 34
F = n_theta + 7

nsamples = sample.shape[0]

theta0 = sample[:, 0:n_theta]
theta1 = sample[:, n_theta:2*n_theta]
sigma  = sample[:, -7]
mu0    = sample[:, -6]
mu1    = sample[:, -5]
tau0   = sample[:, -4]
tau1   = sample[:, -3]
phi0   = sample[:, -2]
phi1   = sample[:, -1]


# cov = np.cov(sample, rowvar=True)
# print(cov)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

axs[0].plot(theta0[:, 0], alpha=0.5, linewidth=1, marker='o', markersize=0., label=r'$\theta_0$')
axs[0].axhline(theta0[:, 0].mean(), alpha=0.5, color='tab:blue')
axs[0].plot(theta0[:, 1], alpha=0.5, linewidth=1, marker='o', markersize=0., label=r'$\theta_1$')
axs[0].axhline(theta0[:, 1].mean(), alpha=0.5, color='tab:orange')
axs[0].plot(theta0[:, 5], alpha=0.5, linewidth=1, marker='o', markersize=0., label=r'$\theta_5$')
axs[0].axhline(theta0[:, 5].mean(), alpha=0.5, color='tab:green')

axs[1].plot(theta1[:, 0], alpha=0.5, linewidth=1, marker='o', markersize=0., label=r'$\theta_0$')
axs[1].axhline(theta1[:, 0].mean(), alpha=0.5, color='tab:blue')
axs[1].plot(theta1[:, 1], alpha=0.5, linewidth=1, marker='o', markersize=0., label=r'$\theta_1$')
axs[1].axhline(theta1[:, 1].mean(), alpha=0.5, color='tab:orange')
axs[1].plot(theta1[:, 5], alpha=0.5, linewidth=1, marker='o', markersize=0., label=r'$\theta_5$')
axs[1].axhline(theta1[:, 5].mean(), alpha=0.5, color='tab:green')

axs[2].plot(tau0, alpha=0.5, linewidth=1, marker='o', markersize=0., label=r'$\tau_0$')
axs[2].axhline(tau0.mean(), alpha=0.5, color='tab:blue')
axs[2].plot(tau1, alpha=0.5, linewidth=1, marker='o', markersize=0., label=r'$\tau_1$')
axs[2].axhline(tau1.mean(), alpha=0.5, color='tab:orange')

axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.show()

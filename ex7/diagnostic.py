
import argparse

import matplotlib.pyplot as plt
import numpy as np

import util


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

axs[0].plot(theta0[:, 0][-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\theta_0$')
axs[0].axhline(theta0[:, 0].mean(), alpha=0.5, color='tab:blue')
axs[0].plot(theta0[:, 1][-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\theta_1$')
axs[0].axhline(theta0[:, 1].mean(), alpha=0.5, color='tab:orange')
axs[0].plot(theta0[:, 5][-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\theta_5$')
axs[0].axhline(theta0[:, 5].mean(), alpha=0.5, color='tab:green')

axs[1].plot(theta1[:, 0][-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\theta_0$')
axs[1].axhline(theta1[:, 0].mean(), alpha=0.5, color='tab:blue')
axs[1].plot(theta1[:, 1][-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\theta_1$')
axs[1].axhline(theta1[:, 1].mean(), alpha=0.5, color='tab:orange')
axs[1].plot(theta1[:, 5][-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\theta_5$')
axs[1].axhline(theta1[:, 5].mean(), alpha=0.5, color='tab:green')

axs[2].plot(tau0[-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\tau_0$')
axs[2].axhline(tau0.mean(), alpha=0.5, color='tab:blue')
axs[2].plot(tau1[-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\tau_1$')
axs[2].axhline(tau1.mean(), alpha=0.5, color='tab:orange')

axs[0].legend()
axs[1].legend()
axs[2].legend()




ess = util.ess(sample).astype(int)

string = str(sample[:, :n_theta].mean(axis=0)).replace('\n', '\n             ')
print(f'Mean theta0: {string}')
string = str(sample[:, n_theta:2*n_theta].mean(axis=0)).replace('\n', '\n             ')
print(f'     theta1: {string}')
print(f'     sigma : {sample[:, -7].mean(axis=0)}')
print(f'     mu0   : {sample[:, -6].mean(axis=0)}')
print(f'     mu1   : {sample[:, -5].mean(axis=0)}')
print(f'     tau0  : {sample[:, -4].mean(axis=0)}')
print(f'     tau1  : {sample[:, -3].mean(axis=0)}')
print(f'     phi0  : {sample[:, -2].mean(axis=0)}')
print(f'     phi1  : {sample[:, -1].mean(axis=0)}')
print()

string = str(ess[:n_theta]).replace('\n', '\n             ')
print(f'ESS  theta0: {string}')
string = str(ess[n_theta:2*n_theta]).replace('\n', '\n             ')
print(f'     theta1: {string}')
print(f'     sigma : {ess[-7]}')
print(f'     mu0   : {ess[-6]}')
print(f'     mu1   : {ess[-5]}')
print(f'     tau0  : {ess[-4]}')
print(f'     tau1  : {ess[-3]}')
print(f'     phi0  : {ess[-2]}')
print(f'     phi1  : {ess[-1]}')


s_idx_theta0 = np.argsort(ess[:n_theta])
s_idx_theta1 = np.argsort(ess[n_theta:2*n_theta])
s_idx_ess = np.argsort(ess)


fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

axs[0].plot(theta0[:, np.min(s_idx_theta0)][-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\theta_0$')
axs[0].axhline(theta0[:, np.min(s_idx_theta0)].mean(), alpha=0.5, color='tab:blue')

axs[1].plot(theta1[:, np.min(s_idx_theta0)][-1000:], alpha=0.5, linewidth=0, marker='o', markersize=1, label=r'$\theta_0$')
axs[1].axhline(theta1[:, np.min(s_idx_theta0)].mean(), alpha=0.5, color='tab:blue')

axs[0].legend()
axs[1].legend()





plt.show()

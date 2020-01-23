
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from matplotlib.ticker import MaxNLocator

from data import raw_y, y, ind, ids, cnts, is_child, is_child_row

import util

from util import mark_hdi, mark_mean, mark_mode

plt.rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument(dest='input_files', nargs='*',
                    help='''Generate as many samples you want with e.g.
                            (replace N with appropriate number)
                            ```
                                python3 -m simulate -n 10000 -i N
                            ```
                            Analyse with `python -m analyse sample-10000-*.npz`.
                            ''')
args = parser.parse_args()

np.random.seed(1066)

samples = []
for file_name in args.input_files:
    data = np.load(file_name)['data']
    if data.shape[1] == 75:
        samples += [data]
sample = np.concatenate(samples, axis=0)

print(f'Sample shape      : {sample.shape}')

n_theta = 34

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



#
# Figure 0.a
# --- Showing the groups in the data
#

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), constrained_layout=True)

for iRow, row in enumerate(y):
    color = 'salmon' if is_child_row[iRow] else 'gray'
    axs[0].plot(row, color=color, marker='o', linewidth=2.0, markersize=3, alpha=0.4)
    axs[1].plot(np.log(row), color=color, marker='o', linewidth=2.0, markersize=3, alpha=0.4)

axs[0].plot(y[is_child_row, :].max(axis=0), linewidth=1.5, color='salmon', alpha=0.8)
axs[0].plot(y[is_child_row, :].min(axis=0), linewidth=1.5, color='salmon', alpha=0.8)
axs[0].plot(y[is_child_row, :].mean(axis=0), linewidth=1.5, color='red', alpha=0.8)
axs[1].plot(np.log(y[is_child_row, :].max(axis=0)), linewidth=1.5, color='salmon', alpha=0.8)
axs[1].plot(np.log(y[is_child_row, :].min(axis=0)), linewidth=1.5, color='salmon', alpha=0.8)
axs[1].plot(np.log(y[is_child_row, :].mean(axis=0)), linewidth=1.5, color='red', alpha=0.8)

axs[0].plot(y[np.logical_not(is_child_row), :].max(axis=0), linewidth=1.5, color='gray', alpha=0.8)
axs[0].plot(y[np.logical_not(is_child_row), :].min(axis=0), linewidth=1.5, color='gray', alpha=0.8)
axs[0].plot(y[np.logical_not(is_child_row), :].mean(axis=0), linewidth=1.5, color='black', alpha=0.8)
axs[1].plot(np.log(y[np.logical_not(is_child_row), :].max(axis=0)), linewidth=1.5, color='gray', alpha=0.8)
axs[1].plot(np.log(y[np.logical_not(is_child_row), :].min(axis=0)), linewidth=1.5, color='gray', alpha=0.8)
axs[1].plot(np.log(y[np.logical_not(is_child_row), :].mean(axis=0)), linewidth=1.5, color='black', alpha=0.8)

fig.suptitle('Input Data')

axs[0].set_title('data scale')
axs[0].set_ylabel('reaction time [ms]')
axs[0].set_xlabel('attempt')
axs[1].set_title('log scale')
axs[1].set_ylabel('reaction time [log(ms)]')
axs[1].set_xlabel('attempt')

axs[0].set_xlim([0, 19])
axs[1].set_xlim([0, 19])

axs[0].grid()
axs[0].set_axisbelow(True)
axs[1].grid()
axs[1].set_axisbelow(True)

axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))








#
# Figure 0.b
# --- Posterior predicitve distribution
#

# Generation approach 2

N = 10000
nchild = np.sum(is_child_row)
is_child = np.random.choice([0, 1], size=(N,), p=[1 - float(nchild)/n_theta,
                                                  0 + float(nchild)/n_theta])

ppd_sigma = np.random.choice(sigma, size=(N,))

ppd_mu0 = np.random.choice(mu0, size=(N,))
ppd_phi0 = np.random.choice(phi0, size=(N,))
ppd_tau0 = np.random.choice(tau0, size=(N,))
ppd_theta0 = np.random.normal(size=(N,), loc=ppd_mu0+is_child*ppd_phi0, scale=ppd_tau0)

ppd_mu1 = np.random.choice(mu1, size=(N,))
ppd_phi1 = np.random.choice(phi1, size=(N,))
ppd_tau1 = np.random.choice(tau1, size=(N,))
ppd_theta1 = np.random.normal(size=(N,), loc=ppd_mu1+is_child*ppd_phi1, scale=ppd_tau1)

predictive_samples_mixed_2 = np.exp(np.random.normal(size=(N,), loc=ppd_theta0 + ppd_theta1*np.random.choice(10), scale=ppd_sigma))
# predictive_samples_adult_2 = np.exp(np.random.normal(size=(N,), loc=ppd_mu0, scale=ppd_tau0))
# predictive_samples_child_2 = np.exp(np.random.normal(size=(N,), loc=ppd_mu0+ppd_phi0, scale=ppd_tau0))

# Plotting

from scipy.signal import savgol_filter

fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 5), constrained_layout=True)

raw_y_hist, raw_y_bins = np.histogram(raw_y, bins=100, range=(100, 1000), density=True)
yhat = savgol_filter(raw_y_hist, 5, 1)

axs.hist(raw_y, bins=40, range=(100, 1000), density=True, histtype='step', color='black', linestyle='dashed', label=r'$y_{\mathrm{raw}}$')
plt.plot(raw_y_bins[:-1]+(raw_y_bins[1]-raw_y_bins[0])*0.5, yhat, color='black', linewidth=0.75)

axs.hist(predictive_samples_mixed_2, bins=40, range=(100, 1000), density=True, histtype='step', color='steelblue', label=r'$y_{\mathrm{mixed}}$')
mark_mean(axs, predictive_samples_mixed_2, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='steelblue', text_rotation=0)
mark_mode(axs, predictive_samples_mixed_2, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='steelblue', text_rotation=0)

axs.set_title(r'Posterior Predictive Check')
axs.legend()
axs.grid()
axs.set_axisbelow(True)
# axs.set_xlim([5.0, 6.75])
# axs.set_ylim([0., 3.0])
axs.set_xlabel(r'$y$ [ms]')
axs.set_ylabel('density')









plt.show()



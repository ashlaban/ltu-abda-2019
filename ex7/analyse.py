
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
raw_samples = []
samples_ex6 = []
for file_name in args.input_files:
    npz = np.load(file_name)
    data = npz['data']
    if data.shape[1] == 75:
        samples += [data]
        raw_data = npz['raw_data']
        raw_samples += [raw_data]
    else:
        samples_ex6 += [data]

if samples:
    sample     = np.concatenate(samples, axis=0)
    raw_sample = np.concatenate(raw_samples, axis=0)
else:
    sample     = np.empty(shape=(0, 75), dtype=np.float64)
    raw_sample = np.empty(shape=(0, 75), dtype=np.float64)

if samples_ex6:
    sample_ex6 = np.concatenate(samples_ex6, axis=0)
else:
    sample_ex6 = np.empty(shape=(0, 38), dtype=np.float64)


if samples:
    print(f'Sample shape      : {sample.shape}')
    print(f'Sample (raw) shape: {raw_sample.shape}')
if samples_ex6:
    print(f'Sample (ex6) shape: {sample_ex6.shape}')

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

raw_theta0 = raw_sample[:, 0:n_theta]
raw_theta1 = raw_sample[:, n_theta:2*n_theta]
raw_sigma  = raw_sample[:, -7]
raw_mu0    = raw_sample[:, -6]
raw_mu1    = raw_sample[:, -5]
raw_tau0   = raw_sample[:, -4]
raw_tau1   = raw_sample[:, -3]
raw_phi0   = raw_sample[:, -2]
raw_phi1   = raw_sample[:, -1]

sigma_ex6  = sample_ex6[:, -4]

# #
# # Figure 0.a
# # --- Showing the groups in the data
# #

# fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), constrained_layout=True)

# for iRow, row in enumerate(y):
#     color = 'salmon' if is_child_row[iRow] else 'gray'
#     axs[0].plot(row, color=color, marker='o', linewidth=2.0, markersize=3, alpha=0.4)
#     axs[1].plot(np.log(row), color=color, marker='o', linewidth=2.0, markersize=3, alpha=0.4)

# axs[0].plot(y[is_child_row, :].max(axis=0), linewidth=1.5, color='salmon', alpha=0.8)
# axs[0].plot(y[is_child_row, :].min(axis=0), linewidth=1.5, color='salmon', alpha=0.8)
# axs[0].plot(y[is_child_row, :].mean(axis=0), linewidth=1.5, color='red', alpha=0.8)
# axs[1].plot(np.log(y[is_child_row, :].max(axis=0)), linewidth=1.5, color='salmon', alpha=0.8)
# axs[1].plot(np.log(y[is_child_row, :].min(axis=0)), linewidth=1.5, color='salmon', alpha=0.8)
# axs[1].plot(np.log(y[is_child_row, :].mean(axis=0)), linewidth=1.5, color='red', alpha=0.8)

# axs[0].plot(y[np.logical_not(is_child_row), :].max(axis=0), linewidth=1.5, color='gray', alpha=0.8)
# axs[0].plot(y[np.logical_not(is_child_row), :].min(axis=0), linewidth=1.5, color='gray', alpha=0.8)
# axs[0].plot(y[np.logical_not(is_child_row), :].mean(axis=0), linewidth=1.5, color='black', alpha=0.8)
# axs[1].plot(np.log(y[np.logical_not(is_child_row), :].max(axis=0)), linewidth=1.5, color='gray', alpha=0.8)
# axs[1].plot(np.log(y[np.logical_not(is_child_row), :].min(axis=0)), linewidth=1.5, color='gray', alpha=0.8)
# axs[1].plot(np.log(y[np.logical_not(is_child_row), :].mean(axis=0)), linewidth=1.5, color='black', alpha=0.8)

# fig.suptitle('Input Data')

# axs[0].set_title('data scale')
# axs[0].set_ylabel('reaction time [ms]')
# axs[0].set_xlabel('attempt')
# axs[1].set_title('log scale')
# axs[1].set_ylabel('reaction time [log(ms)]')
# axs[1].set_xlabel('attempt')

# axs[0].set_xlim([0, 19])
# axs[1].set_xlim([0, 19])

# axs[0].grid()
# axs[0].set_axisbelow(True)
# axs[1].grid()
# axs[1].set_axisbelow(True)

# axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))









# Figure 0.b
# --- Posterior predicitve distribution
#

# Generation approach 2

N = 10000
ppd_ids = np.random.choice(ids, size=(N,))
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

predictive_samples = np.exp(np.random.normal(size=(N,), loc=np.random.choice(theta0.reshape(-1), size=(N,)) +
                                                            np.random.choice(theta1.reshape(-1), size=(N,))*(np.random.choice(10)+1),
                                                        scale=ppd_sigma))

predictive_samples_group = np.exp(np.random.normal(size=(N,), loc=ppd_theta0 + 
                                                                  ppd_theta1*(np.random.choice(10)+1),
                                                               scale=ppd_sigma))

# Plotting

from scipy.signal import savgol_filter

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 5), constrained_layout=True)

# raw_y_hist, raw_y_bins = np.histogram(raw_y, bins=100, range=(100, 800), density=True)
# yhat = savgol_filter(raw_y_hist, 5, 1)

axs.hist(raw_y, bins=100, range=(100, 800), density=True, histtype='step', color='black', linestyle='dashed', label=r'$y_{\mathrm{raw}}$')
mark_mean(axs, raw_y, hist_kwargs=dict(range=(100, 800), bins=100, density=True), text_color='black', text_rotation=0)
mark_mode(axs, raw_y, hist_kwargs=dict(range=(100, 800), bins=100, density=True), text_color='black', text_rotation=0)
# plt.plot(raw_y_bins[:-1]+(raw_y_bins[1]-raw_y_bins[0])*0.5, yhat, color='black', linewidth=0.75)

axs.hist(predictive_samples_group, bins=100, range=(100, 800), density=True, histtype='step', color='steelblue', label=r'$y_{\mathrm{group}}$')
mark_mean(axs, predictive_samples_group, hist_kwargs=dict(range=(100, 800), bins=100, density=True), text_color='steelblue', text_rotation=0)
mark_mode(axs, predictive_samples_group, hist_kwargs=dict(range=(100, 800), bins=100, density=True), text_color='steelblue', text_rotation=0)

axs.hist(predictive_samples, bins=100, range=(100, 800), density=True, histtype='step', color='red', label=r'$y_{\mathrm{ind}}$')
mark_mean(axs, predictive_samples, hist_kwargs=dict(range=(100, 800), bins=100, density=True), text_color='red', text_rotation=0)
mark_mode(axs, predictive_samples, hist_kwargs=dict(range=(100, 800), bins=100, density=True), text_color='red', text_rotation=0)

axs.set_title(r'Posterior Predictive Check')
axs.legend()
axs.grid()
axs.set_axisbelow(True)
# axs.set_xlim([5.0, 6.75])
# axs.set_ylim([0., 3.0])
axs.set_xlabel(r'$y$ [ms]')
axs.set_ylabel('density')





#
# Figure 1
# --- Distribution for expected reaction time for first, and fifth attempts
#

ind0_attempt1 = np.exp(theta0[:, 0] + theta1[:, 0]*1 + sigma**2/2)
ind0_attempt5 = np.exp(theta0[:, 0] + theta1[:, 0]*5 + sigma**2/2)

ind2_attempt1 = np.exp(theta0[:, 2] + theta1[:, 2]*1 + sigma**2/2)
ind2_attempt5 = np.exp(theta0[:, 2] + theta1[:, 2]*5 + sigma**2/2)

ind3_attempt1 = np.exp(theta0[:, 3] + theta1[:, 3]*1 + sigma**2/2)
ind3_attempt5 = np.exp(theta0[:, 3] + theta1[:, 3]*5 + sigma**2/2)

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(10, 5), constrained_layout=True)

hist_kwargs = dict(bins=100, density=True)

axs[0, 0].hist(ind0_attempt1, **hist_kwargs, histtype='step', color='steelblue', label=r'$y_{\mathrm{mixed}}$')
mark_hdi(axs[0, 0], ind0_attempt1, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[0, 0], ind0_attempt1, hist_kwargs=hist_kwargs, text_color='steelblue', text_rotation=0)
axs[0, 0].axvline(x=y[0, 0], color='steelblue')
axs[0, 0].hist(ind0_attempt5, **hist_kwargs, histtype='step', color='tab:red', label=r'$y_{\mathrm{mixed}}$')
mark_hdi(axs[0, 0], ind0_attempt5, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[0, 0], ind0_attempt5, hist_kwargs=hist_kwargs, text_color='tab:red', text_rotation=0)
axs[0, 0].axvline(x=y[0, 4], color='tab:red')
axs[0, 0].set_xlabel(r'Individual 0')

axs[0, 1].hist(ind2_attempt1, **hist_kwargs, histtype='step', color='steelblue', label=r'$y_{\mathrm{mixed}}$')
mark_hdi(axs[0, 1], ind2_attempt1, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[0, 1], ind2_attempt1, hist_kwargs=hist_kwargs, text_color='steelblue', text_rotation=0)
axs[0, 1].axvline(x=y[2, 0], color='steelblue')
axs[0, 1].hist(ind2_attempt5, **hist_kwargs, histtype='step', color='tab:red', label=r'$y_{\mathrm{mixed}}$')
mark_hdi(axs[0, 1], ind2_attempt5, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[0, 1], ind2_attempt5, hist_kwargs=hist_kwargs, text_color='tab:red', text_rotation=0)
axs[0, 1].axvline(x=y[2, 4], color='tab:red')
axs[0, 1].set_xlabel(r'Individual 2')

axs[0, 2].hist(ind3_attempt1, **hist_kwargs, histtype='step', color='steelblue', label=r'$y_{\mathrm{mixed}}$')
mark_hdi(axs[0, 2], ind3_attempt1, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[0, 2], ind3_attempt1, hist_kwargs=hist_kwargs, text_color='steelblue', text_rotation=0)
axs[0, 2].axvline(x=y[3, 0], color='steelblue')
axs[0, 2].hist(ind3_attempt5, **hist_kwargs, histtype='step', color='tab:red', label=r'$y_{\mathrm{mixed}}$')
mark_hdi(axs[0, 2], ind3_attempt5, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[0, 2], ind3_attempt5, hist_kwargs=hist_kwargs, text_color='tab:red', text_rotation=0)
axs[0, 2].axvline(x=y[3, 4], color='tab:red')
axs[0, 2].set_xlabel(r'Individual 3')


x = np.linspace(0, 20, 100).reshape(-1, 1)

N = 100
idx = np.random.choice(nsamples, size=(N,))

axs[1, 0].plot(x, np.exp(theta0[idx, 0] + theta1[idx, 0]*x + sigma[idx]**2/2), color='tab:blue', alpha=0.1)
axs[1, 1].plot(x, np.exp(theta0[idx, 2] + theta1[idx, 2]*x + sigma[idx]**2/2), color='tab:blue', alpha=0.1)
axs[1, 2].plot(x, np.exp(theta0[idx, 3] + theta1[idx, 3]*x + sigma[idx]**2/2), color='tab:blue', alpha=0.1)

axs[1, 0].plot(range(1, len(y[0, :])+1), y[0, :], color='tab:red', marker='o', markersize=3, linewidth=0)
axs[1, 1].plot(range(1, len(y[2, :])+1), y[2, :], color='black', marker='o', markersize=3, linewidth=0)
axs[1, 2].plot(range(1, len(y[3, :])+1), y[3, :], color='black', marker='o', markersize=3, linewidth=0)

axs[1, 0].axvline(x=1, color='steelblue')
axs[1, 0].axvline(x=5, color='tab:red')
axs[1, 1].axvline(x=1, color='steelblue')
axs[1, 1].axvline(x=5, color='tab:red')
axs[1, 2].axvline(x=1, color='steelblue')
axs[1, 2].axvline(x=5, color='tab:red')


axs[1, 0].set_xlim([0, 20])
axs[1, 1].set_xlim([0, 20])
axs[1, 2].set_xlim([0, 20])
axs[1, 0].set_ylim([100, 800])
axs[1, 1].set_ylim([100, 800])
axs[1, 2].set_ylim([100, 800])


#
# Diagnostics plots
#
#

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5, 5), constrained_layout=True)

hist_kwargs = dict(range=(0.125, .175), bins=100, density=True)
axs.hist(sigma, **hist_kwargs, histtype='step', color='steelblue', label=r'$\sigma_{\mathrm{ex7}}$')
mark_hdi(axs, sigma, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs, sigma, hist_kwargs=hist_kwargs, text_color='steelblue', text_rotation=0)
axs.set_xlabel(r'$\sigma$')

if samples_ex6:
    axs.hist(sigma_ex6, **hist_kwargs, histtype='step', color='tab:red', label=r'$\sigma_{\mathrm{ex6}}$')
    mark_hdi(axs, sigma_ex6, hist_kwargs=hist_kwargs, text_rotation=90)
    mark_mean(axs, sigma_ex6, hist_kwargs=hist_kwargs, text_color='tab:red', text_rotation=0)

fig.legend()


#
# Diagnostics plots
#
#

fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(5, 10), constrained_layout=True)

hist_kwargs = dict(range=(0.125, .175), bins=100, density=True)
axs[0].hist(sigma, **hist_kwargs, histtype='step', color='steelblue', label=r'$\sigma_{\mathrm{ex7}}$')
mark_hdi(axs[0], sigma, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[0], sigma, hist_kwargs=hist_kwargs, text_color='steelblue', text_rotation=0)
axs[0].set_xlabel(r'$\sigma$')

if samples_ex6:
    axs[0].hist(sigma_ex6, **hist_kwargs, histtype='step', color='tab:red', label=r'$\sigma_{\mathrm{ex6}}$')
    mark_hdi(axs[0], sigma_ex6, hist_kwargs=hist_kwargs, text_rotation=90)
    mark_mean(axs[0], sigma_ex6, hist_kwargs=hist_kwargs, text_color='tab:red', text_rotation=0)

hist_kwargs = dict(range=(0.1, 0.3), bins=100, density=True)
axs[1].hist(tau0, **hist_kwargs, histtype='step', color='steelblue', label=r'$\tau_0$')
mark_hdi(axs[1], tau0, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[1], tau0, hist_kwargs=hist_kwargs, text_color='steelblue', text_rotation=0)
axs[1].set_xlabel(r'$\tau_0$')

hist_kwargs = dict(range=(0., 0.75), bins=100, density=True)
axs[2].hist(phi0, **hist_kwargs, histtype='step', color='steelblue', label=r'$\phi_0$')
mark_hdi(axs[2], phi0, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[2], phi0, hist_kwargs=hist_kwargs, text_color='steelblue', text_rotation=0)
axs[2].set_xlabel(r'$\phi_0$')

hist_kwargs = dict(range=(-.075, 0.), bins=100, density=True)
axs[3].hist(phi1, **hist_kwargs, histtype='step', color='steelblue', label=r'$\phi_1$')
mark_hdi(axs[3], phi1, hist_kwargs=hist_kwargs, text_rotation=90)
mark_mean(axs[3], phi1, hist_kwargs=hist_kwargs, text_color='steelblue', text_rotation=0)
axs[3].set_xlabel(r'$\phi_1$')

fig.legend()

plt.show()



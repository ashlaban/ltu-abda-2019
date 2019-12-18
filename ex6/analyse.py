
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
samples_ex5 = []
for file_name in args.input_files:
    data = np.load(file_name)['data']
    if data.shape[1] == 38:
        samples += [data]
    else:
        samples_ex5 += [data]
sample = np.concatenate(samples, axis=0)
sample_ex5 = np.concatenate(samples_ex5, axis=0) if samples_ex5 else np.empty(shape=(0, 37),
                                                                              dtype=np.float)

print(f'Sample shape      : {sample.shape}')
print(f'Sample shape (ex5): {sample_ex5.shape}')

n_theta = 34

nsamples = sample.shape[0]
nsamples5 = sample_ex5.shape[0]

theta = sample[:, 0:n_theta]
sigma = sample[:, -4]
mu = sample[:, -3]
tau = sample[:, -2]
phi = sample[:, -1]


theta5 = sample_ex5[:, 0:n_theta]
sigma5 = sample_ex5[:, -3]
mu5 = sample_ex5[:, -2]
tau5 = sample_ex5[:, -1]




#
# Figure 0
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
# Figure 1
# --- Effect size
#

fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(8, 5), constrained_layout=True)

N = 100000
nchild = np.sum(is_child_row)
is_child = np.random.choice([0, 1], size=(N,), p=[1 - float(nchild)/n_theta,
                                                  0 + float(nchild)/n_theta])
row_idx = np.random.choice(nsamples, size=(N,))

mixed_ids = np.arange(n_theta)
adult_ids = np.arange(n_theta)[np.logical_not(is_child_row)]
child_ids = np.arange(n_theta)[is_child_row]

iTheta_mixed = np.random.choice(mixed_ids, size=(N,))
iTheta_adult = np.random.choice(adult_ids, size=(N,))
iTheta_child = np.random.choice(child_ids, size=(N,))

theta_mixed = theta[row_idx, iTheta_mixed]
theta_adult = theta[row_idx, iTheta_adult]
theta_child = theta[row_idx, iTheta_child]

theta_mixed_gen = np.random.normal(size=(N,), loc=mu[row_idx] + is_child*phi[row_idx], scale=tau[row_idx])
theta_adult_gen = np.random.normal(size=(N,), loc=mu[row_idx], scale=tau[row_idx])
theta_child_gen = np.random.normal(size=(N,), loc=mu[row_idx] + phi[row_idx], scale=tau[row_idx])

# This definition mirrors the Krushke model (y = N(theta + phi, sigma)).
theta_diff = theta_child - theta_adult
theta_diff_mean = theta_diff.mean()
theta_diff = (theta_diff - theta_diff_mean) * sigma.mean() + theta_diff_mean

theta_diff_gen = theta_child_gen - theta_adult_gen
theta_diff_gen_mean = theta_diff_gen.mean()
theta_diff_gen = (theta_diff_gen - theta_diff_gen_mean) * sigma.mean() + theta_diff_gen_mean

axs[0].hist(theta_diff, bins=100, density=True, histtype='step', color='black', linestyle='solid',
                    label=r'$\phi$ posterior')
axs[0].hist(theta_diff_gen, bins=100, density=True, histtype='step', color='black', linestyle='dashed',
                    label=r'$\phi$ prior')

mark_mean(axs[0], theta_diff_gen, hist_kwargs=dict(bins=100, density=True), text_xoffset=0.)

# mark_mean(axs[0], theta_diff, hist_kwargs=dict(bins=100, density=True), text_ha='center')
# mark_mode(axs[0], theta_diff, hist_kwargs=dict(bins=100, density=True), text_ha='center')
# mark_hdi(axs[0], theta_diff,
#          hist_kwargs=dict(bins=100, density=True),
#          line_kwargs=dict(color='black', linewidth=0.5),
#          text_yoffset=0.02, text_va='bottom')

# mark_mean(axs[0], theta_diff_gen,
#           hist_kwargs=dict(bins=100, density=True),
#           marker_kwargs=dict(marker='^', fillstyle='none'),
#           text_ha='center')
# mark_hdi(axs[0], theta_diff_gen,
#           hist_kwargs=dict(bins=100, density=True),
#           line_kwargs=dict(color='black', linewidth=0.5, linestyle='dashed'),
#           text_yoffset=-0.02, text_va='top', text_ha=['left', 'right'])

_phi = phi[row_idx]
axs[1].hist(_phi, bins=100, density=True, histtype='step', color='black', linestyle='solid',
                    label=r'$\phi$ posterior')

mark_mean(axs[1], _phi, hist_kwargs=dict(bins=100, density=True), text_xoffset=0.)
mark_hdi(axs[1], _phi,
         hist_kwargs=dict(bins=100, density=True),
         line_kwargs=dict(color='black', linewidth=0.5),
         text_yoffset=0., text_va='bottom')

fig.suptitle('Effect Size (log reaction time)')

axs[0].set_title(r'Krushke $\phi = \theta_{\mathrm{child}} - \theta_{\mathrm{adult}}$')
axs[0].legend()
axs[0].set_xlabel(r'$\phi$')
axs[0].set_ylabel(r'density')
axs[0].set_xlim([0.0, 0.8])
axs[0].grid()
axs[0].set_axisbelow(True)

axs[1].set_title(r'Gelman $\phi$')
axs[1].legend()
axs[1].set_xlabel(r'$\phi$')
axs[1].set_ylabel(r'density')
axs[1].set_xlim([0.0, 0.8])
axs[1].grid()
axs[1].set_axisbelow(True)



#
# Figure 2
# --- Posterior of tau
#

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5), constrained_layout=True)

ax.hist(tau, range=(0.1, 0.5), bins=100, density=True, histtype='step', color='black', label=r'$\tau$')
ax.hist(tau5, range=(0.1, 0.5), bins=100, density=True, histtype='step', color='black', linestyle='dashed', label=r'$\tau_{\mathrm{ex5}}$')

ax.set_title(r'Posterior distribution ($\tau$)')
ax.legend()
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'density')
ax.grid()
ax.set_axisbelow(True)

mark_mean(ax, tau, hist_kwargs=dict(range=(0.1, 0.5), bins=100, density=True), text_xoffset=0.01)
mark_mode(ax, tau, hist_kwargs=dict(range=(0.1, 0.5), bins=100, density=True), text_xoffset=0.01)
mark_hdi(ax, tau, hist_kwargs=dict(range=(0.1, 0.5), bins=100, density=True),
                  line_kwargs=dict(color='black', linewidth=0.5),
                  text_yoffset=0.3, text_va='bottom')

mark_mean(ax, tau5, hist_kwargs=dict(range=(0.1, 0.5), bins=100, density=True),
                    marker_kwargs=dict(marker='^', fillstyle='none'),
                    text_xoffset=0.01)
mark_mode(ax, tau5, hist_kwargs=dict(range=(0.1, 0.5), bins=100, density=True),
                    marker_kwargs=dict(marker='^', fillstyle='none'),
                    text_xoffset=0.01)
mark_hdi(ax, tau5, hist_kwargs=dict(range=(0.1, 0.5), bins=100, density=True),
                   line_kwargs=dict(color='black', linewidth=0.5, linestyle='dashed'),
                   text_yoffset=-0.3, text_va='top', text_ha=['left', 'right'])







#
# Figure 0.1
# --- Posterior of theta
#

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 5), constrained_layout=True)


N = 100000
nchild = np.sum(is_child_row)
is_child = np.random.choice([0, 1], size=(N,), p=[1 - float(nchild)/n_theta,
                                                  0 + float(nchild)/n_theta])
row_idx = np.random.choice(nsamples, size=(N,))

mixed_ids = np.arange(n_theta)
adult_ids = np.arange(n_theta)[np.logical_not(is_child_row)]
child_ids = np.arange(n_theta)[is_child_row]

iTheta_mixed = np.random.choice(mixed_ids, size=(N,))
iTheta_adult = np.random.choice(adult_ids, size=(N,))
iTheta_child = np.random.choice(child_ids, size=(N,))

theta_mixed = theta[row_idx, iTheta_mixed]
theta_adult = theta[row_idx, iTheta_adult]
theta_child = theta[row_idx, iTheta_child]

theta_mixed_gen = np.random.normal(size=(N,), loc=mu[row_idx] + is_child*phi[row_idx], scale=tau[row_idx])
theta_adult_gen = np.random.normal(size=(N,), loc=mu[row_idx], scale=tau[row_idx])
theta_child_gen = np.random.normal(size=(N,), loc=mu[row_idx] + phi[row_idx], scale=tau[row_idx])

axs[0].hist(theta_adult_gen, bins=100, density=True, histtype='step', color='black', linestyle='solid', label=r'$\theta_{\mathrm{adult}}$ generated')
axs[0].hist(theta_adult, bins=100, density=True, histtype='step', color='black', linestyle='dashed', label=r'$\theta_{\mathrm{adult}}$ posterior')

mark_mean(axs[0], theta_adult_gen, hist_kwargs=dict(bins=100, density=True), text_xoffset=0.)
mark_mode(axs[0], theta_adult_gen, hist_kwargs=dict(bins=100, density=True), text_yoffset=-0.15)
mark_hdi(axs[0], theta_adult_gen, hist_kwargs=dict(bins=100, density=True),
                                  line_kwargs=dict(color='black', linewidth=0.5),
                                  text_yoffset=0., text_va='bottom')

axs[1].hist(theta_child_gen, bins=100, density=True, histtype='step', color='red', linestyle='solid', label=r'$\theta_{\mathrm{child}}$ generated')
axs[1].hist(theta_child, bins=100, density=True, histtype='step', color='red', linestyle='dashed', label=r'$\theta_{\mathrm{child}}$ posterior')

mark_mean(axs[1], theta_child_gen, hist_kwargs=dict(bins=100, density=True))
mark_mode(axs[1], theta_child_gen, hist_kwargs=dict(bins=100, density=True), text_yoffset=0.1)
mark_hdi(axs[1], theta_child_gen, hist_kwargs=dict(bins=100, density=True),
                                  line_kwargs=dict(color='red', linewidth=0.5),
                                  text_yoffset=0., text_va='bottom')

mark_mean(axs[1], theta_child, hist_kwargs=dict(bins=100, density=True),
                               marker_kwargs=dict(fillstyle='none', marker='^'))
mark_mode(axs[1], theta_child, hist_kwargs=dict(bins=100, density=True),
                               marker_kwargs=dict(fillstyle='none', marker='^'), text_yoffset=0.1)

axs[2].hist(theta_mixed_gen, bins=100, density=True, histtype='step', color='steelblue', linestyle='solid', label=r'$\theta_{\mathrm{mixed}}$ generated')
axs[2].hist(theta_mixed, bins=100, density=True, histtype='step', color='steelblue', linestyle='dashed', label=r'$\theta_{\mathrm{mixed}}$ posterior')

mark_mean(axs[2], theta_mixed_gen, hist_kwargs=dict(bins=100, density=True))
mark_mode(axs[2], theta_mixed_gen, hist_kwargs=dict(bins=100, density=True))
mark_hdi(axs[2], theta_mixed_gen, hist_kwargs=dict(bins=100, density=True),
                                  line_kwargs=dict(color='steelblue', linewidth=0.5),
                                  text_yoffset=0., text_va='bottom')

fig.suptitle(r'Posterior vs. Prior ($\theta$)')
axs[0].set_title('Adult')
axs[0].legend()
axs[0].grid()
axs[0].set_axisbelow(True)
axs[0].set_xlim([5.0, 6.75])
axs[0].set_ylim([0., 3.0])
axs[0].set_xlabel(r'$\theta$')
axs[0].set_ylabel('density')

axs[1].set_title('Child')
axs[1].legend()
axs[1].grid()
axs[1].set_axisbelow(True)
axs[1].set_xlim([5.0, 6.75])
axs[1].set_ylim([0., 3.0])
axs[1].set_xlabel(r'$\theta$')
axs[1].set_ylabel('density')

axs[2].set_title('Mixed')
axs[2].legend()
axs[2].grid()
axs[2].set_axisbelow(True)
axs[2].set_xlim([5.0, 6.75])
axs[2].set_ylim([0., 3.0])
axs[2].set_xlabel(r'$\theta$')
axs[2].set_ylabel('density')





#
# Figure 3
# --- Prior of log(expected reaction time)
#

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5), constrained_layout=True)

nchild = np.sum(is_child_row)
is_child = np.random.choice([0, 1], size=(len(phi),), p=[1 - float(nchild)/n_theta,
                                                         0 + float(nchild)/n_theta])

theta_mixed_prior_samples = np.random.normal(loc=mu + is_child*phi, scale=tau)
theta_adult_prior_samples = np.random.normal(loc=mu, scale=tau)
theta_child_prior_samples = np.random.normal(loc=mu + phi, scale=tau)

theta_ex5_prior_samples = np.random.normal(loc=mu5, scale=tau5)

ax.hist(theta_mixed_prior_samples, bins=100, density=True, histtype='step', color='steelblue', label=r'$\theta_{\mathrm{mixed}}$')
ax.hist(theta_adult_prior_samples, bins=100, density=True, histtype='step', color='black', label=r'$\theta_{\mathrm{adult}}$')
ax.hist(theta_child_prior_samples, bins=100, density=True, histtype='step', color='red', label=r'$\theta_{\mathrm{child}}$')
ax.hist(theta_ex5_prior_samples, bins=100, density=True, histtype='step', color='steelblue', linestyle='dashed', label=r'$\theta_{\mathrm{ex5}}$')

mark_mean(ax, theta_mixed_prior_samples, hist_kwargs=dict(bins=100, density=True), text_color='steelblue')
mark_hdi(ax, theta_mixed_prior_samples, hist_kwargs=dict(bins=100, density=True),
                                        line_kwargs=dict(color='steelblue', linewidth=0.5),
                                        text_yoffset=0., text_va='bottom')

mark_mean(ax, theta_adult_prior_samples, hist_kwargs=dict(bins=100, density=True), text_color='black')
mark_hdi(ax, theta_adult_prior_samples, hist_kwargs=dict(bins=100, density=True),
                                        line_kwargs=dict(color='black', linewidth=0.5),
                                        text_yoffset=0., text_va='bottom')

mark_mean(ax, theta_child_prior_samples, hist_kwargs=dict(bins=100, density=True), text_color='red')
mark_hdi(ax, theta_child_prior_samples, hist_kwargs=dict(bins=100, density=True),
                                        line_kwargs=dict(color='red', linewidth=0.5),
                                        text_yoffset=0., text_va='bottom')

ax.set_title(r'Prior distribution ($\theta$)')
ax.legend()
ax.grid()
ax.set_axisbelow(True)
ax.set_xlim([5.0, 6.75])
ax.set_ylim([0., 3.0])
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('density')
ax.legend()



#
# Figure 4
# --- Posterior predicitve distribution
#


# Generation approach 1

N = 100000
mixed_ids = np.arange(n_theta)
adult_ids = np.arange(n_theta)[np.logical_not(is_child_row)]
child_ids = np.arange(n_theta)[is_child_row]
iTheta_mixed = np.random.choice(mixed_ids, size=(N,))
iTheta_adult = np.random.choice(adult_ids, size=(N,))
iTheta_child = np.random.choice(child_ids, size=(N,))

row_idx = np.random.choice(nsamples, size=(N,))
row_idx5 = np.random.choice(nsamples5, size=(N,))

predictive_samples_mixed_1 = np.exp(np.random.normal(theta[row_idx, iTheta_mixed], sigma[row_idx]))
predictive_samples_adult_1 = np.exp(np.random.normal(theta[row_idx, iTheta_adult], sigma[row_idx]))
predictive_samples_child_1 = np.exp(np.random.normal(theta[row_idx, iTheta_child], sigma[row_idx]))

predictive_samples_mixed5_1 = np.exp(np.random.normal(theta5[row_idx5, iTheta_mixed], sigma5[row_idx5]))



# Generation approach 2

N = 100000
nchild = np.sum(is_child_row)
is_child = np.random.choice([0, 1], size=(N,), p=[1 - float(nchild)/n_theta,
                                                  0 + float(nchild)/n_theta])
ppd_mu = np.random.choice(mu, size=(N,))
ppd_phi = np.random.choice(phi, size=(N,))
ppd_tau = np.random.choice(tau, size=(N,))
ppd_theta = np.exp(np.random.normal(size=(N,), loc=ppd_mu+is_child*ppd_phi, scale=ppd_tau))

predictive_samples_mixed_2 = np.exp(np.random.normal(size=(N,), loc=ppd_mu+is_child*ppd_phi, scale=ppd_tau))
predictive_samples_adult_2 = np.exp(np.random.normal(size=(N,), loc=ppd_mu, scale=ppd_tau))
predictive_samples_child_2 = np.exp(np.random.normal(size=(N,), loc=ppd_mu+ppd_phi, scale=ppd_tau))

ppd_mu5 = np.random.choice(mu5, size=(N,))
ppd_tau5 = np.random.choice(tau5, size=(N,))
predictive_samples_mixed5_2 = np.exp(np.random.normal(size=(N,), loc=ppd_mu5, scale=ppd_tau5))

# Plotting

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), constrained_layout=True)

axs[0].hist(predictive_samples_adult_1, bins=100, range=(100, 1000), density=True, histtype='step', color='black', label=r'$y_{\mathrm{adult}}$')
axs[0].hist(predictive_samples_child_1, bins=100, range=(100, 1000), density=True, histtype='step', color='red', label=r'$y_{\mathrm{child}}$')
axs[0].hist(predictive_samples_mixed_1, bins=100, range=(100, 1000), density=True, histtype='step', color='steelblue', label=r'$y_{\mathrm{mixed}}$')
axs[0].hist(predictive_samples_mixed5_1, bins=100, range=(100, 1000), density=True, histtype='step', color='steelblue', linestyle='dashed', label=r'$y_{\mathrm{ex5}}$')

mark_mean(axs[0], predictive_samples_adult_1, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='black', text_rotation=0)
mark_mean(axs[0], predictive_samples_child_1, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='red', text_rotation=0)
mark_mean(axs[0], predictive_samples_mixed_1, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='steelblue', text_rotation=0)
mark_mode(axs[0], predictive_samples_adult_1, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='black', text_rotation=0)
mark_mode(axs[0], predictive_samples_child_1, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='red', text_rotation=0)
mark_mode(axs[0], predictive_samples_mixed_1, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='steelblue', text_rotation=0)

axs[1].hist(predictive_samples_adult_2, bins=100, range=(100, 1000), density=True, histtype='step', color='black', label=r'$y_{\mathrm{adult}}$')
axs[1].hist(predictive_samples_child_2, bins=100, range=(100, 1000), density=True, histtype='step', color='red', label=r'$y_{\mathrm{child}}$')
axs[1].hist(predictive_samples_mixed_2, bins=100, range=(100, 1000), density=True, histtype='step', color='steelblue', label=r'$y_{\mathrm{mixed}}$')
axs[1].hist(predictive_samples_mixed5_2, bins=100, range=(100, 1000), density=True, histtype='step', color='steelblue', linestyle='dashed', label=r'$y_{\mathrm{ex5}}$')

mark_mean(axs[1], predictive_samples_adult_2, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='black', text_rotation=0)
mark_mean(axs[1], predictive_samples_child_2, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='red', text_rotation=0)
mark_mean(axs[1], predictive_samples_mixed_2, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='steelblue', text_rotation=0)
mark_mode(axs[1], predictive_samples_adult_2, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='black', text_rotation=0)
mark_mode(axs[1], predictive_samples_child_2, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='red', text_rotation=0)
mark_mode(axs[1], predictive_samples_mixed_2, hist_kwargs=dict(range=(100, 1000), bins=100, density=True), text_color='steelblue', text_rotation=0)

fig.suptitle(r'Posterior predictive distribution')
axs[0].set_title(r'Method 1: Sample $\theta$ from posterior')
axs[0].legend()
axs[0].grid()
axs[0].set_axisbelow(True)
# axs[0].set_xlim([5.0, 6.75])
# axs[0].set_ylim([0., 3.0])
axs[0].set_xlabel(r'$y$ [ms]')
axs[0].set_ylabel('density')

axs[1].set_title(r'Method 2: Generate $\theta$ from prior')
axs[1].legend()
axs[1].grid()
axs[1].set_axisbelow(True)
# axs[1].set_xlim([5.0, 6.75])
# axs[1].set_ylim([0., 3.0])
axs[1].set_xlabel(r'$y$ [ms]')
axs[1].set_ylabel('density')

plt.show()

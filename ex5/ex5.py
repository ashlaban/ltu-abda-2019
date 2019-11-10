
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from ex5_data import raw_y, y, ind, ids, cnts

import util

plt.rc('text', usetex=True)

sample_0 = np.load('sample-10000-0.npz')['data']
sample_1 = np.load('sample-10000-1.npz')['data']
sample_2 = np.load('sample-10000-2.npz')['data']
sample_3 = np.load('sample-10000-3.npz')['data']
sample_4 = np.load('sample-10000-4.npz')['data']
sample_5 = np.load('sample-10000-5.npz')['data']
sample_6 = np.load('sample-10000-6.npz')['data']
sample_7 = np.load('sample-10000-7.npz')['data']
sample_8 = np.load('sample-10000-8.npz')['data']
sample_9 = np.load('sample-10000-9.npz')['data']
sample = np.concatenate([sample_0, sample_1, sample_2, sample_3, sample_4,
                         sample_5, sample_6, sample_7, sample_8, sample_9],
                        axis=0)

n_theta = 34

theta = sample[:, 0:n_theta]
sigma = sample[:, -3]
mu = sample[:, -2]
tau = sample[:, -1]



predictive_dist = scipy.stats.norm
predictive_samples = []
for iTheta in range(34):
    predictive_samples += [np.exp(scipy.stats.norm.rvs(theta[:, iTheta], sigma))]
predictive_samples = np.stack(predictive_samples)
print(predictive_samples.shape)




#
# Figure 0
# --- Data exploration
#

fig, axs = plt.subplots(ncols=2, nrows=1)

for row in y:
    axs[0].plot(row, marker='o', linewidth=0.5, markersize=1)
    axs[1].plot(np.log(row), marker='o', linewidth=0.5, markersize=1)

axs[0].plot(y.max(axis=0), linewidth=1.5, color='black', alpha=0.8)
axs[0].plot(y.mean(axis=0), linewidth=1.5, color='black', alpha=0.8)
axs[0].plot(y.min(axis=0), linewidth=1.5, color='black', alpha=0.8)
axs[1].plot(np.log(y.max(axis=0)), linewidth=1.5, color='black', alpha=0.8)
axs[1].plot(np.log(y.mean(axis=0)), linewidth=1.5, color='black', alpha=0.8)
axs[1].plot(np.log(y.min(axis=0)), linewidth=1.5, color='black', alpha=0.8)

axs[0].set_ylabel('Reaction time [ms]')
axs[0].set_xlabel('attempt')
axs[1].set_xlabel('attempt')

axs[0].set_xlim([0, 20])
axs[1].set_xlim([0, 20])

axs[0].grid()
axs[1].grid()
plt.tight_layout()





#
# Figure 3a
# --- Compare expected reaction times
#

plt.figure(figsize=(3,6))

n_theta_eff = int(n_theta)
# n_theta_eff = int(n_theta/2)
# n_theta_eff = 2
hs = []
x = None
for iTheta in range(n_theta_eff):
    h, x = np.histogram(np.exp(theta[:, iTheta] + sigma**2), bins=100, range=(150, 700), density=True)
    hs += [h]
h = np.stack(hs, axis=1)
x = np.concatenate([x[:-2], x[-1:]])

y_mean = np.mean(y, axis=1)[:n_theta_eff]
mean = np.mean(np.exp(theta + sigma.reshape(-1, 1)**2), axis=0)
idx = np.argsort(mean[:n_theta_eff])
h = h[:, idx]

ticks = np.arange(n_theta_eff)*0.015

# Plot expected response time of all
plt.hlines(ticks, xmin=150, xmax=700, alpha=0.8, linewidth=0.5, color='gray')
plt.step(x, ticks+h, color='tab:blue', linewidth=0.5)
plt.vlines(x=mean[idx], ymin=ticks, ymax=ticks+0.015, color='black', linewidth=0.5)
plt.vlines(x=y_mean[idx], ymin=ticks, ymax=ticks+0.015, color='gray', linewidth=0.5)

# Plot the dude
idx_dude = np.argwhere(idx==3)[0]
print(idx_dude)
plt.step(x, 0.015*idx_dude+h[:, idx_dude], color='tab:red', linewidth=0.5)

plt.yticks(ticks, idx)
plt.xlim([150, 700])

plt.title('Expected reaction time\nper individual comparision')
plt.xlabel('Expected reacation time $[ms]$')
plt.ylabel('Individual')
plt.tight_layout()




#
# Figure 3b
# --- Compare expected log reaction times
#

plt.figure(figsize=(3,6))

n_theta_eff = int(n_theta)
# n_theta_eff = int(n_theta/2)
# n_theta_eff = 2
hs = []
x = None
for iTheta in range(n_theta_eff):
    h, x = np.histogram(theta[:, iTheta], bins=100, range=(5.2, 6.6), density=True)
    hs += [h]
h = np.stack(hs, axis=1)
x = np.concatenate([x[:-2], x[-1:]])

y_mean = np.mean(np.log(y), axis=1)
mean = np.mean(theta, axis=0)
idx = np.argsort(mean[:n_theta_eff])
h = h[:, idx]


ticks = np.arange(n_theta_eff)*8.0

plt.hlines(ticks, xmin=5.2, xmax=6.6, alpha=0.8, linewidth=0.5, color='gray')
plt.step(x, ticks+h, color='tab:blue', linewidth=0.5)
plt.vlines(x=mean[idx], ymin=ticks, ymax=ticks+8.0, color='black', linewidth=0.5)
plt.vlines(x=y_mean[idx], ymin=ticks, ymax=ticks+8.0, color='gray', linewidth=0.5)

# Plot the dude
idx_dude = np.argwhere(idx==3)[0]
print(idx_dude)
plt.step(x, 8.0*idx_dude+h[:, idx_dude], color='tab:red', linewidth=0.5)

plt.yticks(ticks, idx)
plt.xlim([5.2, 6.6])

plt.title('Model parameter $\\theta$ comparision\nper individual')
plt.xlabel('$\\theta$')
plt.ylabel('Individual')
plt.tight_layout()


#
# Figure 1
# --- Expected reation time for individual 3 (0-indexed)
#

plt.figure(figsize=(6.4, 4.8/2))
expected_reacation_time = np.exp(theta[:, 3] + sigma**2 / 2)
hdi_low, hdi_high = util.hdi(expected_reacation_time)
h, bins = np.histogram(expected_reacation_time, bins=100, density=True)

plt.hlines(0, xmin=bins[0], xmax=bins[-2], alpha=0.8, linewidth=0.5, color='gray')
plt.step(bins[:-1], h, linewidth=0.5, color='tab:red')
plt.vlines(x=expected_reacation_time.mean(), ymin=0, ymax=h[np.digitize(expected_reacation_time.mean(), bins)], color='black', linewidth=0.5)
plt.vlines(x=np.mean(y[3, :]), ymin=0, ymax=h[np.digitize(np.mean(y[3, :]), bins)], color='gray', linewidth=0.5)
plt.vlines(x=hdi_low, ymin=0, ymax=h[np.digitize(hdi_low, bins)], color='black', linewidth=0.5, linestyle='dashed')
plt.vlines(x=hdi_high, ymin=0, ymax=h[np.digitize(hdi_high, bins)], color='black', linewidth=0.5, linestyle='dashed')

plt.annotate(f'{int(np.mean(y[3, :]))}', (np.mean(y[3, :]), h[np.digitize(np.mean(y[3, :]), bins)]), color='gray', ha='right')
plt.annotate(f'{int(expected_reacation_time.mean())}', (expected_reacation_time.mean(), h[np.digitize(expected_reacation_time.mean(), bins)]))
plt.annotate(f'{int(hdi_low)}', (hdi_low, h[np.digitize(hdi_low, bins)]))
plt.annotate(f'{int(hdi_high)}', (hdi_high, h[np.digitize(hdi_high, bins)]))

plt.xlim([bins[0], bins[-2]])

plt.title('Expected reacation time for individual 3: "the dude"')
plt.xlabel('Expected reacation time $[ms]$')
plt.ylabel('density')
plt.tight_layout()

print('mean:', expected_reacation_time.mean())
print('hdi:', (hdi_low , hdi_high))




#
# Figure 2.i
# --- Expected group means
#

plt.figure(figsize=(6.4, 4.8/2))
expected_reacation_time_group = np.exp(mu + tau**2 / 2 + sigma**2 / 2)
hdi_low, hdi_high = util.hdi(expected_reacation_time_group)
h, bins = np.histogram(expected_reacation_time_group, bins=100, range=(0, 800), density=True)

plt.hlines(0, xmin=bins[0], xmax=bins[-2], alpha=0.8, linewidth=0.5, color='gray')
plt.step(bins[:-1], h, linewidth=0.5, color='tab:blue')
plt.vlines(x=expected_reacation_time_group.mean(), ymin=0, ymax=h[np.digitize(expected_reacation_time_group.mean(), bins)], color='black', linewidth=0.5)
plt.vlines(x=np.mean(y[3, :]), ymin=0, ymax=h[np.digitize(np.mean(y[3, :]), bins)], color='gray', linewidth=0.5)
plt.vlines(x=hdi_low, ymin=0, ymax=h[np.digitize(hdi_low, bins)], color='black', linewidth=0.5, linestyle='dashed')
plt.vlines(x=hdi_high, ymin=0, ymax=h[np.digitize(hdi_high, bins)], color='black', linewidth=0.5, linestyle='dashed')

plt.annotate(f'{int(np.mean(y))}', (np.mean(y), h[np.digitize(np.mean(y[3, :]), bins)]), color='gray', ha='right')
plt.annotate(f'{int(expected_reacation_time_group.mean())}', (expected_reacation_time_group.mean(), h[np.digitize(expected_reacation_time_group.mean(), bins)]))
plt.annotate(f'{int(hdi_low)}', (hdi_low, h[np.digitize(hdi_low, bins)]))
plt.annotate(f'{int(hdi_high)}', (hdi_high, h[np.digitize(hdi_high, bins)]))

plt.xlim([bins[0], bins[-2]])

plt.title('Expected reacation time for the group')
plt.xlabel('Expected reacation time $[ms]$')
plt.ylabel('density')
plt.tight_layout()




#
# Figure 2.ii
# --- Predictive posterior
#

plt.figure(figsize=(6.4, 4.8/2))

mean = predictive_samples.mean()
hdi_low, hdi_high = util.hdi(predictive_samples.reshape(-1))

hraw, bins = np.histogram(raw_y, bins=50, range=(0, 800), density=True)
h, _ = np.histogram(predictive_samples.reshape(-1), bins=50, range=(0, 800), density=True)

plt.step(bins[:-1], hraw, color='tab:blue', linewidth=0.5, linestyle='dashed')
plt.step(bins[:-1], h, linewidth=0.5, color='tab:blue')

plt.vlines(x=mean, ymin=0, ymax=h[np.digitize(mean, bins)], color='black', linewidth=0.5)
plt.vlines(x=np.mean(y), ymin=0, ymax=h[np.digitize(np.mean(y), bins)], color='gray', linewidth=0.5)
plt.vlines(x=hdi_low, ymin=0, ymax=h[np.digitize(hdi_low, bins)], color='black', linewidth=0.5, linestyle='dashed')
plt.vlines(x=hdi_high, ymin=0, ymax=h[np.digitize(hdi_high, bins)], color='black', linewidth=0.5, linestyle='dashed')

plt.annotate(f'{int(np.mean(y))}', (np.mean(y), h[np.digitize(np.mean(y), bins)]), color='gray', ha='right')
plt.annotate(f'{int(mean)}', (mean, h[np.digitize(mean, bins)]))
plt.annotate(f'{int(hdi_low)}', (hdi_low, h[np.digitize(hdi_low, bins)]))
plt.annotate(f'{int(hdi_high)}', (hdi_high, h[np.digitize(hdi_high, bins)]))

plt.title('Posterior predicted reaction time for a random individual')
plt.xlabel('Reacation time $[ms]$')
plt.ylabel('density')
plt.tight_layout()




plt.show()

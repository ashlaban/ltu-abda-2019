
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.stats

import samplers
import util

plt.rc('text', usetex=True)

class Bernoulli(object):
    @staticmethod
    def sample(shape=(1,), t=0.5):
        '''`t` is probability of a 1 in output.'''
        return (np.random.uniform(size=shape) > (1-t)).astype(np.int)
    @staticmethod
    def pdf(x, theta=0.5):
        return theta**x * (1-theta)**(1-x)
    @staticmethod
    def logpdf(x, theta=0.5):
        return x*np.log(theta) + (1-x)*np.log(1-theta)
    @staticmethod
    def likelihood(seq, theta=0.5):
        theta = np.asarray(theta)
        ret = np.prod([Bernoulli.pdf(x, theta) for x in seq], axis=0)
        if ret.shape == ():
            ret = ret.reshape(1)
        ret[np.logical_or(theta<0, theta>1)] = 0
        return ret
    @staticmethod
    def loglikelihood(seq, theta=0.5):
        theta = np.asarray(theta)
        ret = np.sum([Bernoulli.logpdf(x, theta) for x in seq], axis=0)
        if ret.shape == ():
            ret = ret.reshape(1)
        ret[np.logical_or(theta<0, theta>1)] = -100000
        return ret

class Beta(object):
    @staticmethod
    def sample(shape=(1,)):
        return Slice.sample(shape, pdf)
    @staticmethod
    def pdf(theta, a=1, b=1):
        return theta**(a-1) * (1-theta)**(b-1) / _beta(a, b)
    @staticmethod
    def logpdf(theta, a=1, b=1):
        return np.log(Beta.pdf(theta, a, b))
    @staticmethod
    def likelihood(seq, a=1, b=1):
        return np.prod([Beta.pdf(x, a, b) for x in seq])
    @staticmethod
    def loglikelihood(seq, a=1, b=1):
        return np.sum([Beta.logpdf(x, a, b) for x in seq])


y = np.asarray([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
z = np.asarray([1, 0, 0, 0, 0, 0, 0, 1, 1, 0])

theta = np.linspace(0., 1., 100, endpoint=True)
theta_diff = np.linspace(-1., 1., 200, endpoint=True)
likelihood_y = Bernoulli.likelihood(y, theta)
likelihood_z = Bernoulli.likelihood(z, theta)
posterior_y = likelihood_y/np.sum(likelihood_y)/0.01
posterior_z = likelihood_z/np.sum(likelihood_z)/0.01

pdf_y = lambda theta: Bernoulli.likelihood(y, theta)
pdf_z = lambda theta: Bernoulli.likelihood(z, theta)
logpdf_y = lambda theta: Bernoulli.loglikelihood(y, theta)
logpdf_z = lambda theta: Bernoulli.loglikelihood(z, theta)

# Using fast sampler (for small dimensionality)
sample_y = samplers.Naive.sample(int(1e6), logpdf_y, ylim=-6, log=True)
sample_z = samplers.Naive.sample(int(1e6), logpdf_z, ylim=-5, log=True)

# Similar results but slower:
# sample_y = samplers.Slice.sample(int(1e4), logpdf_y, step_size=[1])
# sample_z = samplers.Slice.sample(int(1e4), logpdf_z, step_size=[1])

_y, _x = np.histogram(sample_y-sample_z, bins=30, range=[-1., 1.], density=True)
posterior_diff_interp = scipy.interpolate.interp1d(_x[:-1], _y,
                                                   kind='zero',
                                                   fill_value='extrapolate')
posterior_diff_interp_linear = scipy.interpolate.interp1d(np.diff(_x)/2 + _x[:-1], _y,
                                                          kind='linear',
                                                          fill_value='extrapolate')
posterior_diff = posterior_diff_interp(theta_diff)

hdi_sample_y = util.hdi(sample_y)
hdi_sample_z = util.hdi(sample_z)
hdi_sample_diff = util.hdi(sample_y-sample_z)

mean_y = np.mean(sample_y)
mean_z = np.mean(sample_z)
mean_diff = np.mean(sample_y-sample_z)

mode_y = theta[np.argmax(posterior_y)]
mode_z = theta[np.argmax(posterior_z)]
mode_diff = theta_diff[np.argmax(posterior_diff)]

# probablility theta > 0.5
p = np.sum(posterior_y[theta>0.5])*0.01

# probablility theta_y > theta_z
p_y_lgt_z = np.sum(posterior_diff[theta_diff>0.0])*0.01

print('mean_y:', mean_y)
print('mean_z:', mean_z)
print('mean_diff:', mean_diff)

print('mode_y:', mode_y)
print('mode_z:', mode_z)
print('mode_diff:', mode_diff)

print('hdi_y:', hdi_sample_y)
print('hdi_z:', hdi_sample_z)
print('hdi_diff:', hdi_sample_diff)

print('p:', p)
print('p_y_lgt_z:', p_y_lgt_z)

# ============================================================================
# === Only plotting below
# ============================================================================

# ============================================================================
# === Figure 1
# ============================================================================

plt.figure()

plt.plot(theta, posterior_y, color='tab:blue')
plt.hist(sample_y, bins=30, range=[0, 1],
         density=True, color='tab:blue', alpha=0.2)
plt.fill_between(theta[50:], y1=posterior_y[50:], y2=0,
         linestyle='None', hatch='//', facecolor='None', edgecolor='tab:blue', alpha=0.5)
plt.plot([hdi_sample_y[0], hdi_sample_y[0]],
         [0, Bernoulli.likelihood(y, hdi_sample_y[0])/np.sum(likelihood_y)/0.01],
         color='black')
plt.plot([hdi_sample_y[1], hdi_sample_y[1]],
         [0, Bernoulli.likelihood(y, hdi_sample_y[1])/np.sum(likelihood_y)/0.01],
         color='black')

plt.plot(theta, posterior_z, color='tab:orange')
plt.hist(sample_z, bins=30, range=[0, 1],
         density=True, color='tab:orange', alpha=0.2)

# Annotate y
plt.text(mode_y, posterior_y[np.digitize(mode_y, theta)]+0.025,
         f'$\\mathrm{{Mode}}_y = {mode_y:.3}$', ha='right')
plt.text(mean_y, posterior_y[np.digitize(mean_y, theta)]-0.025,
         f'$\\mathrm{{Mean}}_y = {mean_y:.3}$', ha='right')
plt.annotate(f'$\\mathrm{{HDI}}\\ (0.95) = ({hdi_sample_y[0]:.3}, {hdi_sample_y[1]:.3})$',
            xy=(hdi_sample_y[0] + (hdi_sample_y[1]-hdi_sample_y[0])/2,
                Bernoulli.likelihood(y, hdi_sample_y[0])/np.sum(likelihood_y)/0.01 - 0.1),
            ha='center', va='center')
plt.annotate('',
            xy=(hdi_sample_y[0], posterior_y[np.digitize(hdi_sample_y[0], theta)]/2), xycoords='data',
            xytext=(hdi_sample_y[1], posterior_y[np.digitize(hdi_sample_y[0], theta)]/2), textcoords='data',
            arrowprops=dict(arrowstyle="-"),
            )
plt.text(0.5, 2.5, f'\\noindent$p(\\theta>0.5|y)\\\\={p:.4}$')

# Annotate z
plt.text(mode_z, posterior_z[np.digitize(mode_z, theta)]+0.025,
         f'$\\mathrm{{Mode}}_z = {mode_z:.3}$', ha='center')

plt.title('Distribution of $\\theta_y$ and $\\theta_z$ (Sample size $N=10^6$)')
plt.xlabel('$\\theta$')
plt.ylabel('$p(\\theta | y\\ \\mathrm{{or}}\\ z)$')

# ============================================================================
# === Figure 2
# ============================================================================

plt.figure()
hist, bins = np.histogram(sample_y-sample_z, bins=30, range=[-1, 1], density=True)
bins = np.diff(bins)/2 + bins[:-1]

to_minimize = lambda x: np.sum((scipy.stats.beta.pdf(theta_diff, x[0], x[1]) - posterior_diff_interp_linear(theta_diff))**2)
minimized_params = scipy.optimize.minimize(to_minimize, (1, 1))


plt.plot(theta_diff, scipy.stats.beta.pdf(theta_diff, minimized_params.x[0], minimized_params.x[1]),
         color='tab:blue', linestyle='dashed')
plt.fill_between(x=np.linspace(0, 1, 1000),
                 y1=posterior_diff_interp(np.linspace(0, 1, 1000),),
                 y2=0,
                 linestyle='None', hatch='//', facecolor='None',
                 edgecolor='tab:blue', alpha=0.5)
plt.hist(sample_y-sample_z, bins=30, range=[-1, 1],
         density=True, color='tab:blue', alpha=0.2)
plt.plot([hdi_sample_diff[0], hdi_sample_diff[0]],
         [0, posterior_diff_interp(hdi_sample_diff[0])],
         color='black')
plt.plot([hdi_sample_diff[1], hdi_sample_diff[1]],
         [0, posterior_diff_interp(hdi_sample_diff[1])],
         color='black')

# Annotate
plt.text(0.15, 1.5,
         f'$\\beta(a={minimized_params.x[0]:.3}; b={minimized_params.x[1]:.3})$', ha='right')
plt.text(mode_diff, posterior_diff_interp(mode_diff)+0.01,
         f'$\\mathrm{{Mode}}_{{\\Delta\\theta}} = {mode_diff:.3}$', ha='right')
plt.annotate(f'$\\mathrm{{HDI}}\\ (0.95) = ({hdi_sample_diff[0]:.3}, {hdi_sample_diff[1]:.3})$',
            xy=(hdi_sample_diff[0]-0.05,
                posterior_diff_interp(hdi_sample_diff[0]) + 0.1),
            ha='right', va='center')
plt.annotate('',
            xy=(hdi_sample_diff[0], posterior_diff_interp(hdi_sample_diff[0])/2), xycoords='data',
            xytext=(hdi_sample_diff[1], posterior_diff_interp(hdi_sample_diff[0])/2), textcoords='data',
            arrowprops=dict(arrowstyle="-"),
            )

plt.title('Distribution of $\\Delta\\theta = \\theta_y - \\theta_z$ (Sample size $N=10^6$)')
plt.xlabel('$\\theta$')
plt.ylabel('$p(\\Delta\\theta| y, z)$')

plt.show()

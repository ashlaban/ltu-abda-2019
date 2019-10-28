
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.stats

from scipy.special import beta as _beta
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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

#
# === Task B 5.a.iii
#
x = np.linspace(0.001, 0.999, 100)
h = 17
t = 3
y = [1]*h + [0]*t

log_post_a = lambda x: (Beta.logpdf(x, a=100, b=100) +
                        Bernoulli.loglikelihood(y, x))
log_post_b = lambda x: (Beta.logpdf(x, a=18.25, b=6.75) +
                        Bernoulli.loglikelihood(y, x))
log_post_c = lambda x: (Beta.logpdf(x, a=1, b=1) +
                        Bernoulli.loglikelihood(y, x))

sample_a = samplers.Naive.sample(shape=int(1e4), pdf=log_post_a, log=True, ylim=-10)
sample_b = samplers.Naive.sample(shape=int(1e4), pdf=log_post_b, log=True, ylim=-7)
sample_c = samplers.Naive.sample(shape=int(1e4), pdf=log_post_c, log=True, ylim=-7)

fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True)

axs[0, 0].set_title('High belief in prior\n$p(\\theta)=\\beta(100,100)$')
axs[0, 1].set_title('Balanced\n$p(\\theta)=\\beta(18.25,6.75)$')
axs[0, 2].set_title('Low belief in prior\n$p(\\theta)=\\beta(1,1)$')

axs[0, 0].plot(x, np.exp(Beta.logpdf(x, a=100, b=100)))
axs[0, 0].set_ylabel('$p(\\theta)$')
axs[1, 0].plot(x, np.exp(Bernoulli.loglikelihood(y, x)))
axs[1, 0].set_ylabel('$p(x|\\theta)$')
axs[2, 0].plot(x, np.exp(Beta.logpdf(x, a=117, b=103)))
axs[2, 0].hist(sample_a, bins=30, range=[0, 1], density=True,
               color='tab:blue', alpha=0.2)
axs[2, 0].set_ylabel('$p(\\theta|x)$')



axs[0, 1].plot(x, np.exp(Beta.logpdf(x, a=18.25, b=6.75)))
axs[1, 1].plot(x, np.exp(Bernoulli.loglikelihood(y, x)))
axs[2, 1].plot(x, np.exp(Beta.logpdf(x, a=35.25, b=9.75)))
axs[2, 1].hist(sample_b, bins=30, range=[0, 1], density=True,
               color='tab:blue', alpha=0.2)



axs[0, 2].plot(x, np.exp(Beta.logpdf(x, a=1, b=1)))
axs[1, 2].plot(x, np.exp(Bernoulli.loglikelihood(y, x)))
axs[2, 2].plot(x, np.exp(Beta.logpdf(x, a=1+h, b=1+t)))
axs[2, 2].hist(sample_c, bins=30, range=[0, 1], density=True,
               color='tab:blue', alpha=0.2)
axs[2, 2].set_xlabel('$\\theta$')


axs[0, 0].set_ylim([0, 12])
axs[0, 1].set_ylim([0, 12])
axs[0, 2].set_ylim([0, 12])

axs[1, 0].set_ylim([0, 0.00025])
axs[1, 1].set_ylim([0, 0.00025])
axs[1, 2].set_ylim([0, 0.00025])

axs[2, 0].set_ylim([0, 12])
axs[2, 1].set_ylim([0, 12])
axs[2, 2].set_ylim([0, 12])

[ax.xaxis.set_minor_locator(AutoMinorLocator(2)) for ax in axs.flatten()]
[ax.yaxis.set_minor_locator(AutoMinorLocator(2)) for ax in axs.flatten()]
[ax.grid() for ax in axs.flatten()]
[ax.grid(which='minor') for ax in axs.flatten()]
plt.tight_layout()



plt.show()
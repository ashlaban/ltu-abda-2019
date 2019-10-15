import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.special import beta as _beta
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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
        return np.prod([Bernoulli.pdf(x, theta) for x in seq])
    @staticmethod
    def loglikelihood(seq, theta=0.5):
        return np.sum([Bernoulli.logpdf(x, theta) for x in seq])




#
# === Task B.2
#
x = [0, 1]
plt.bar(x, [Bernoulli.pdf(X, theta=0.25) for X in x],
        label='$\\theta=0.25$', hatch='\\',
        facecolor='none', edgecolor='tab:blue',
        linewidth=2)
plt.bar(x, [Bernoulli.pdf(X, theta=0.50) for X in x],
        label='$\\theta=0.50$', hatch='/',
        facecolor='none', edgecolor='tab:orange',
        linewidth=2)
plt.xticks([0, 1], ['$\\theta=0$', '$\\theta=1$'])
plt.ylabel('$p(x|\\theta)$')
plt.legend()
plt.grid()
plt.tight_layout()





#
# === Task B.3
# 
plt.figure()
theta = np.linspace(0, 1, 100)
plt.plot(theta, Bernoulli.pdf(1, theta),
         label='$p(x=1|\\theta)$')
plt.plot(theta, Bernoulli.pdf(0, theta),
         label='$p(x=0|\\theta)$')
plt.xlabel('$\\theta$')
plt.ylabel('$p(x|\\theta)$')
plt.legend()
plt.grid()
plt.tight_layout()



#
# === Task B 4.a
# Likelihood for seqs. w. large n -> tends to zero (hence the log-prob trick)
# Diminishes as ~ t**-n
# 
print('Likelihood n=10:', Bernoulli.likelihood([0]*10))
print('Likelihood n=1000:', Bernoulli.likelihood([0]*1000))
print('Likelihood n=100000:', Bernoulli.likelihood([0]*100000))



#
# === Task B 4.c:
# 
print('Log-likelihood n=10:', Bernoulli.loglikelihood([0]*10))
print('Log-likelihood n=1000:', Bernoulli.loglikelihood([0]*1000))
print('Log-likelihood n=100000:', Bernoulli.loglikelihood([0]*100000))





#
# === Task B.4.d
#
x = np.linspace(0.001, 0.999, 100)
plt.figure()
plt.plot(x, np.exp([Bernoulli.loglikelihood([1], X)
                    for X in x]),
         label='$p(x=[1]|\\theta)$')
plt.plot(x, np.exp([Bernoulli.loglikelihood([1,1], X)
                    for X in x]),
         label='$p(x=[1,1]|\\theta)$')
plt.plot(x, np.exp([Bernoulli.loglikelihood([1,1,0,1],
                                            X)
                    for X in x]),
         label='$p(x=[1,1,0,1]|\\theta)$')
plt.xlabel('$\\theta$')
plt.ylabel('$p(x|\\theta)$')
plt.legend()
plt.grid()
plt.tight_layout()


class Beta(object):
    # @staticmethod
    # def sample(shape=(1,), t=0.5):
    #     '''`t` is probability of a 1 in output.'''
    #     return (np.random.uniform(size=shape) > (1-t)).astype(np.int)
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
# === Task B 5.a.ii.1-2:
#
# Posterior distributions given a beta prior
# Difference to previous plot: These are normalised (i.e. integrates to 1).
x = np.linspace(0.001, 0.999, 100)
plt.figure()
plt.plot(x, np.exp([Beta.logpdf(X, a=2, b=1)
                    for X in x]),
         label='$p(\\theta|x=[1])$')
plt.plot(x, np.exp([Beta.logpdf(X, a=3, b=1)
                    for X in x]),
         label='$p(\\theta|x=[1,1])$')
plt.plot(x, np.exp([Beta.logpdf(X, a=4, b=2)
                    for X in x]),
         label='$p(\\theta|x=[1,1,0,1])$')
plt.plot(x, np.exp([Bernoulli.loglikelihood([1], X)
                    for X in x]),
         linestyle='dashed', color='tab:blue')
plt.plot(x, np.exp([Bernoulli.loglikelihood([1,1], X)
                    for X in x]),
         linestyle='dashed', color='tab:orange')
plt.plot(x, np.exp([Bernoulli.loglikelihood([1,1,0,1],
                                            X)
                    for X in x]),
         linestyle='dashed', color='tab:green')
plt.xlabel('$\\theta$')
plt.ylabel('$p(\\theta|x)$')
plt.legend()
plt.grid()
plt.tight_layout()

#
# === Task B 5.a.iii
#
x = np.linspace(0.001, 0.999, 100)
h = 17
t = 3
y = [1]*h + [0]*t
fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True)

axs[0, 0].set_title('High belief in prior\n$p(\\theta)=\\beta(100,100)$')
axs[0, 1].set_title('Balanced\n$p(\\theta)=\\beta(18.25,6.75)$')
axs[0, 2].set_title('Low belief in prior\n$p(\\theta)=\\beta(1,1)$')

axs[0, 0].fill_between(x, np.exp([Beta.logpdf(X, a=100, b=100)
                    for X in x]))
axs[0, 0].set_ylabel('$p(\\theta)$')
axs[1, 0].fill_between(x, np.exp([Bernoulli.loglikelihood(y, X)
                    for X in x]))
axs[1, 0].set_ylabel('$p(x|\\theta)$')
axs[2, 0].fill_between(x, np.exp([Beta.logpdf(X, a=117, b=103)
                    for X in x]))
axs[2, 0].set_ylabel('$p(\\theta|x)$')



axs[0, 1].fill_between(x, np.exp([Beta.logpdf(X, a=18.25, b=6.75)
                    for X in x]))
axs[1, 1].fill_between(x, np.exp([Bernoulli.loglikelihood(y, X)
                    for X in x]))
axs[2, 1].fill_between(x, np.exp([Beta.logpdf(X, a=35.25, b=9.75)
                    for X in x]))



axs[0, 2].fill_between(x, np.exp([Beta.logpdf(X, a=1, b=1)
                    for X in x]))
axs[1, 2].fill_between(x, np.exp([Bernoulli.loglikelihood(y, X)
                    for X in x]))
axs[2, 2].fill_between(x, np.exp([Beta.logpdf(X, a=1+h, b=1+t)
                    for X in x]))
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

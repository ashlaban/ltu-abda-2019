
import numpy as np
import scipy.stats

from numba import jit, vectorize

#
# Model:
#   log y = z
#   z     ~ p(z | theta, sigma)
#   theta ~ p(theta | mu, tau)
#   mu    ~ p(mu)
#   tau   ~ p(tau)
#   sigma ~ p(sigma)
#
# Thus:
#     p(mu, tau, theta, sigma | z)
#         = p(z | theta, sigma) p(theta | mu, tau) p(sigma) p(mu) p(tau)
#         / p(z)
#
# This means we can sample the space (mu, tau, sigma, theta).
# Q: How do we plot  e.g. p(sigma | z)?
# A: We marginalise the posterior, i.e. histogram w.r. only t. sigma.
#

@jit(nopython=True, nogil=True)
def uniform(x, low, high):
    return np.logical_and((x >= low), (x <= high)) * 1./(high-low)

# @jit(nopython=True, nogil=True)
@vectorize()
def loguniform(x, low, high):
    if (x > low) and (x < high):
        return 0
    else:
        return -np.inf
    # x = np.asarray([x]).reshape(-1)
    # mask = np.logical_not(np.logical_and((x > low), (x < high)))
    # ret = np.zeros_like(x)
    # ret[mask] = -np.inf
    # return ret


@jit(nopython=True, nogil=True)
def normal(x, mu, sigma):
    var = sigma**2
    c = 1./(np.sqrt(2*np.pi*var))
    arg = -(x - mu)**2 / (2*var)
    return c*np.exp(arg)

# @jit(nopython=True, nogil=True)
@vectorize
def lognormal(x, mu, sigma):
    if sigma > 0:
        c = sigma
        arg = (x - mu)/sigma
        return -np.log(c) - arg*arg/2
    else:
        return -np.inf
        # return np.ones_like(x)*(-np.inf)


@jit(nopython=True, nogil=True)
def p_mu(mu): return uniform(mu, -100, 100)
@jit(nopython=True, nogil=True)
def p_tau(tau): return uniform(tau, 0.00001, 100)
@jit(nopython=True, nogil=True)
def p_sigma(sigma): return uniform(sigma, 0.00001, 100)
@jit(nopython=True, nogil=True)
def p_theta(theta, mu, tau): return normal(theta, mu, tau)
@jit(nopython=True, nogil=True)
def p_z(z, theta, sigma): return normal(z, theta, sigma)

@jit(nopython=True, nogil=True)
def logp_mu(mu): return np.log(p_mu(mu))
@jit(nopython=True, nogil=True)
def logp_tau(tau): return np.log(p_tau(tau))
@jit(nopython=True, nogil=True)
def logp_sigma(sigma): return np.log(p_sigma(sigma))
@jit(nopython=True, nogil=True)
def logp_theta(theta, mu, tau): return np.log(p_theta(theta, mu, tau))
@jit(nopython=True, nogil=True)
def logp_z(z, theta, sigma): return np.log(p_z(z, theta, sigma))

@jit(nopython=True, nogil=True)
def posterior_probability(z, theta, sigma, mu, tau, id):
    # z can be vector 1d
    # theta can be vector 1d
    # id is mapping, which theta to use for the given z
    _pz = p_z(z, theta[id], sigma)
    _pt = p_theta(theta, mu, tau)
    return (np.prod(_pz) * np.prod(_pt) * p_sigma(sigma) *
            p_mu(mu) * p_tau(tau))

@jit(nopython=True, nogil=True)
def posterior_log_probability(z, theta, sigma, mu, tau, id):
    # z can be vector 1d
    # theta can be vector 1d
    # id is mapping, which theta to use for the given z
    _pz = lognormal(z, theta[id], sigma)
    _pt = lognormal(theta, mu, tau)
    return (np.sum(_pz) + np.sum(_pt) +
            loguniform(sigma, 0, 100) +
            loguniform(mu, -100, 100) +
            loguniform(tau, 0, 100))

@jit(nopython=True, nogil=True)
def posterior_log_probability_old(z, theta, sigma, mu, tau, id):
    # z can be vector 1d
    # theta can be vector 1d
    # id is mapping, which theta to use for the given z
    _pz = logp_z(z, theta[id], sigma)
    _pt = logp_theta(theta, mu, tau)
    return (np.sum(_pz) + np.sum(_pt) + logp_sigma(sigma) +
            logp_mu(mu) + logp_tau(tau))


def gen_sampler_pdf(z, id, n_theta):
    @jit(nopython=True, nogil=True)
    def fn(x):
        theta = x[0:n_theta]
        sigma = x[n_theta]
        mu = x[-2]
        tau = x[-1]
        return posterior_log_probability(z, theta, sigma, mu, tau, id)
    return fn


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    print(f'p_mu(0): {p_mu(0)}, logp_mu(0): {logp_mu(0)}' )
    print(f'p_mu(10000.1): {p_mu(10000.1)}' )

    print(f'p_theta(0, 0, 1): {p_theta(0, 0, 1)}, logp_theta(0, 0, 1): {logp_theta(0, 0, 1)}' )
    print(f'scipy normal(0, 1) at 0: {scipy.stats.norm(loc=0, scale=1).pdf(0)}')

    print(f'p_mu([0, 1]): {p_mu(np.asarray([0, 1]))}, logp_mu([0, 1]): {logp_mu(np.asarray([0, 1]))}' )
    print(f'p_theta([0, 1], 0, 1): {p_theta(np.asarray([0, 1]), 0, 1)}')

    print(posterior_log_probability(np.asarray([0, 1, 0, 1]),
                                    theta=np.asarray([0, 1]),
                                    sigma=1,
                                    mu=0,
                                    tau=1,
                                    id=np.asarray([0, 0, 1, 1])))

    _sampler_pdf = gen_sampler_pdf(z=np.asarray([0, 1, 0, 1]), id=np.asarray([0, 0, 1, 1]), n_theta=2)
    print(_sampler_pdf(x=np.asarray([0, 1, 1, 0, 1])))


    x = np.linspace(-2, 2)
    plt.figure()
    plt.plot(x, lognormal(x, 0, 1), linestyle='solid', color='tab:blue')
    plt.plot(x, np.log(normal(x, 0, 1)), linestyle='dashed', color='tab:blue')
    plt.plot(x, loguniform(x, 0, 1), color='tab:orange')
    

    plt.figure()
    fn1 = list(map(lambda x: posterior_log_probability(np.asarray([np.log(200.)]), np.asarray([x]), 1., 5., 1., np.asarray([0], dtype=np.int)), x))
    fn2 = list(map(lambda x: posterior_log_probability_old(np.asarray([np.log(200.)]), np.asarray([x]), 1., 5., 1., np.asarray([0], dtype=np.int)), x))

    print(fn1)
    print(fn2)
    plt.plot(x, fn1, color='tab:blue')
    plt.plot(x, fn2, color='tab:orange')


    plt.show()


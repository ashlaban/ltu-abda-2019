
import numpy as np
import scipy.stats

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

def uniform(x, low, high):
    return np.logical_and((x >= low), (x <= high)) * 1./(high-low)

# Important: Ensure sigmas of normal distrbutions can't be 0 or lower
def normal(x, mu, sigma):
    var = sigma**2
    c = 1./(np.sqrt(2*np.pi*var))
    arg = -(x - mu)**2 / (2*var)
    return (sigma>0)*c*np.exp(arg)
    # arg = -((x - mu)/2/sigma)**2
    # return np.exp(arg)

def p_mu(mu): return uniform(mu, -100, 100)
def p_tau(tau): return uniform(tau, 0.00001, 100)
def p_sigma(sigma): return uniform(sigma, 0.00001, 100)
def p_theta(theta, mu, tau): return normal(theta, mu, tau)
def p_z(z, theta, sigma): return normal(z, theta, sigma)

def logp_mu(mu): return np.log(p_mu(mu))
def logp_tau(tau): return np.log(p_tau(tau))
def logp_sigma(sigma): return np.log(p_sigma(sigma))
def logp_theta(theta, mu, tau): return np.log(p_theta(theta, mu, tau))
def logp_z(z, theta, sigma): return np.log(p_z(z, theta, sigma))

def posterior_probability(z, theta, sigma, mu, tau, id):
    # z can be vector 1d
    # theta can be vector 1d
    # id is mapping, which theta to use for the given z
    _pz = p_z(z, theta[id], sigma)
    _pt = p_theta(theta, mu, tau)
    return (np.prod(_pz) * np.prod(_pt) * p_sigma(sigma) *
            p_mu(mu) * p_tau(tau))

def posterior_log_probability(z, theta, sigma, mu, tau, id):
    # z can be vector 1d
    # theta can be vector 1d
    # id is mapping, which theta to use for the given z
    _pz = logp_z(z, theta[id], sigma)
    _pt = logp_theta(theta, mu, tau)
    return (np.sum(_pz) + np.sum(_pt) + logp_sigma(sigma) +
            logp_mu(mu) + logp_tau(tau))


def gen_sampler_pdf(z, id, n_theta):
    def fn(x):
        theta = x[0:n_theta]
        sigma = x[n_theta]
        mu = x[-2]
        tau = x[-1]
        return posterior_log_probability(z, theta, sigma, mu, tau, id)
    return fn


if __name__ == '__main__':
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



import numpy as np
import scipy.stats

from numba import jit, vectorize

#
# Model:
#   log y = z
#   z     ~ p(z | theta, sigma)
#   theta ~ p(theta + phi*is_child | mu, tau)
#   sigma ~ p(sigma)
#   mu    ~ p(mu)
#   tau   ~ p(tau)
#
# An alternative is to model z as p(z | theta+phi*is_child, sigma) instead.
# These methods are equivalent but have different interpretations of priors.
#

@jit(nopython=True, nogil=True)
def uniform(x, low, high):
    return np.logical_and((x >= low), (x <= high)) * 1./(high-low)


@jit(nopython=True, nogil=True)
def normal(x, mu, sigma):
    var = sigma**2
    c = 1./(np.sqrt(2*np.pi*var))
    arg = -(x - mu)**2 / (2*var)
    return c*np.exp(arg)


# @jit(nopython=True, nogil=True)
@vectorize()
def loguniform(x, low, high):
    if (x > low) and (x < high):
        return 0
    else:
        return -np.inf


# @jit(nopython=True, nogil=True)
@vectorize
def lognormal(x, mu, sigma):
    if sigma > 0:
        c = sigma
        arg = (x - mu)/sigma
        return -np.log(c) - arg*arg/2
    else:
        return -np.inf


@jit(nopython=True, nogil=True)
def posterior_log_probability(z, theta, sigma, mu, tau, phi, id, child_id):
    _pz = lognormal(z, theta[id], sigma)
    _pt = lognormal(theta, mu + child_id*phi, tau)
    return (np.sum(_pz) + np.sum(_pt) +
            loguniform(sigma, 0, 100) +
            loguniform(mu, -100, 100) +
            loguniform(tau, 0, 100)   +
            loguniform(phi, -100, 100))


def pack(theta, sigma, mu, tau, phi):
    return np.concatenate([theta, sigma, mu, tau, phi])


def unpack(x):
    theta = x[0:n_theta]
    sigma = x[n_theta]
    mu = x[-3]
    tau = x[-2]
    phi = x[-1]
    return theta, sigma, mu, tau, phi


def gen_sampler_pdf(z, id, child_id, n_theta):
    @jit(nopython=True, nogil=True)
    def fn(x):
        theta = x[0:n_theta]
        sigma = x[n_theta]
        mu = x[-3]
        tau = x[-2]
        phi = x[-1]
        return posterior_log_probability(z, theta, sigma, mu, tau, phi, id, child_id)
    return fn


def gen_sampler_initial_x(n_theta):
    return np.ones(shape=(n_theta+4), dtype=np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_theta = 2
    id_data = np.asarray([0, 1], dtype=np.int)
    child_id = np.asarray([0, 1], dtype=np.int)
    z = np.asarray([250, 260], dtype=np.float32)
    pdf = gen_sampler_pdf(z=z, id=id_data, child_id=child_id, n_theta=n_theta)
    x = gen_sampler_initial_x(n_theta=n_theta)
    print(pdf(x))

    assert((lognormal(np.asarray([1, 2]), 1, 1) ==
            lognormal(np.asarray([1, 2]), np.asarray([1, 1]), 1)).all())

    plt.show()


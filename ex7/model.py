
import numpy as np
import scipy.stats

from numba import jit, vectorize

#
# y  : reaction time
# x0 : category, child=1, adult=0
# x1 : attemt no.
#
# Model:
#   log y  = z
#   z      ~ N(theta0 + theta1*x1, sigma)
#   theta0 ~ N(mu0 + phi0*x0, tau0)
#   theta1 ~ N(mu1 + phi1*x0, tau1)
#   sigma  ~ U(0, 100)
#   mu0    ~ U(-100, 100)
#   mu1    ~ U(-100, 100)
#   tau0   ~ U(0, 100)
#   tau1   ~ U(0, 100)
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
def log_model(z, theta0, theta1, sigma, mu0, mu1, tau0, tau1, phi0, phi1,
              id, x0, x1):
    #
    # x0: id category, child=1, adult=0 (for sample category use x0[id])
    # x1: attemt no. (0, \infty)

    # Standard parameterisation
    # _pz  = lognormal(z, theta0[id] + theta1[id]*x1, sigma)
    # _pt0 = lognormal(theta0, mu0 + phi0*x0, tau0)
    # _pt1 = lognormal(theta1, mu1 + phi1*x0, tau1)

    # Reparameterisation
    eta0 = theta0[:]
    eta1 = theta1[:]
    theta0 = mu0 + phi0*x0 + tau0*eta0
    theta1 = mu1 + phi1*x0 + tau1*eta1
    _pz  = lognormal(z, theta0[id] + theta1[id]*x1, sigma)
    _pt0 = lognormal(eta0, 0., 1.)
    _pt1 = lognormal(eta1, 0., 1.)

    return (np.sum(_pz) + np.sum(_pt0) + np.sum(_pt1) +
            loguniform(sigma,    0, 100) +
            # loguniform(mu0  , -100, 100) +
            # loguniform(mu1  , -100, 100) +
            loguniform(tau0 ,    0, 100) +
            loguniform(tau1 ,    0, 100)
            # loguniform(phi0 , -100, 100) +
            # loguniform(phi1 , -100, 100)
           )


def pack(theta0, theta1, sigma, mu0, mu1, tau0, tau1, phi0, phi1):
    return np.concatenate([theta0, theta1, sigma, mu0, mu1, tau0, tau1, phi0,
                           phi1])


def unpack(x, n_theta):
    theta0 = x[0:n_theta]
    theta1 = x[n_theta:2*n_theta]
    sigma  = x[-7]
    mu0    = x[-6]
    mu1    = x[-5]
    tau0   = x[-4]
    tau1   = x[-3]
    phi0   = x[-2]
    phi1   = x[-1]
    # theta0 = mu0 + phi0*x0 + tau0*eta0
    # theta1 = mu1 + phi1*x0 + tau1*eta1
    return (theta0, theta1, sigma, mu0, mu1, tau0, tau1, phi0, phi1)


def gen_sampler_pdf(z, id, x0, x1, n_theta):
    @jit(nopython=True, nogil=True)
    def pdf(x):
        theta0 = x[0:n_theta]
        theta1 = x[n_theta:2*n_theta]
        sigma  = x[-7]
        mu0    = x[-6]
        mu1    = x[-5]
        tau0   = x[-4]
        tau1   = x[-3]
        phi0   = x[-2]
        phi1   = x[-1]
        # theta0 = mu0 + phi0*x0 + tau0*eta0
        # theta1 = mu1 + phi1*x0 + tau1*eta1
        # print(x, theta0, theta1, sigma, mu0, mu1, tau0, tau1, phi0, phi1)
        return log_model(z, theta0, theta1, sigma, mu0, mu1, tau0, tau1, phi0,
                         phi1, id, x0, x1)
    return pdf


def gen_sampler_initial_x(n_theta):
    return np.ones(shape=(2*n_theta+7), dtype=np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_theta = 2
    id_data = np.asarray([0, 1], dtype=np.int)
    child_id = np.asarray([0, 1], dtype=np.int)
    x1 = np.asarray([0, 1], dtype=np.int)
    z = np.asarray([250, 260], dtype=np.float32)
    pdf = gen_sampler_pdf(z=z, id=id_data, x0=child_id, x1=x1, n_theta=n_theta)
    x = np.arange(2*n_theta+7)
    print(pdf(x))

    assert((lognormal(np.asarray([1, 2]), 1, 1) ==
            lognormal(np.asarray([1, 2]), np.asarray([1, 1]), 1)).all())

    plt.show()


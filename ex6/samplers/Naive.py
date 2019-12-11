
import time

import numpy as np

from ._base import SamplerBase

class Naive(SamplerBase):
    '''
    '''

    @staticmethod
    def sample(shape, pdf, log=False, xlim=[-5, 5], ylim=1,
               batch_size=1000, dtype=np.float):
        '''Use uniform rejection sampling in a hypercube.

        A probability distribution function can be approximated by sampling
        uniformly from its input space an accepting samples in proportion to the
        target probability distribution [0].

        More advanced versions can use any proposal distribution as long as
        the ratio between the proposal distribution and the target
        distribution is known, and the (scaled) proposal distribution
        dominates the target distribution. Optimal efficiency is achieved when
        proposal and target distributions are identical.

        Note: Since the sampling efficiency is directly proportional to the
              ratio between the proposal distribution and the target
              distribution convergence can be slow.

        [0]: https://en.wikipedia.org/wiki/Rejection_sampling

        Arguments
        ---------
        shape:
        pdf:

        Keyword Arguments
        -----------------
        log:
        xlim:
        ylim:
        batch_size:
        dtype:

        Returns
        -------
        A vector of shape `shape` sampled from the given pdf.

        '''
        xlim_low, xlim_high = xlim

        orig_shape = shape
        if isinstance(shape, int):
            shape = np.asarray([shape, 1], dtype=np.int)

        nsamples = np.prod(shape)
        samps_remaining = nsamples
        ndims = len(shape)

        # pdf is function that returns the (possibly estimated) pdf at a given position
        # Note, arity of pdf must match ndims
        # inefficient for high dimensionality!

        time_start = time.time()
        batches = []
        ngenerated = 0
        naccepted = 0
        while samps_remaining > 0:
            current_batch_size = [batch_size] + [shape[-1]]

            x = np.random.uniform(xlim_low, xlim_high, size=current_batch_size)
            
            if log:
                rng = ylim - (np.random.exponential(size=batch_size))
            else:
                rng = np.log(np.random.uniform(0, ylim, size=batch_size))

            if log:
                likelihood = np.sum(pdf(x), axis=1)
            else:    
                likelihood = np.sum(np.log(pdf(x)), axis=1)

            accept = rng < likelihood

            batches += [x[accept]]

            ngenerated += batch_size
            naccepted += np.sum(accept)
            samps_remaining -= np.sum(accept)
        time_end = time.time()
        total_time = time_end - time_start

        print(f'Efficiency (naive): {naccepted/ngenerated}')
        print(f'Time taken (naive): {total_time:.5}s ({total_time/nsamples:.5}s per sample)')
        return np.concatenate(batches).reshape(-1)[:nsamples].reshape(orig_shape)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import scipy.stats

    pdf = scipy.stats.norm.pdf

    # ======
    # === 1D
    # ======

    N = 10000
    sample_naive = Naive.sample([N, 1], pdf, xlim=[-5, 5], ylim=pdf(0))
    sample_direct = np.random.normal(size=[N, 1])
    print('sample_naive.shape:', sample_naive.shape)
    print('sample_direct.shape:', sample_direct.shape)

    fig = plt.figure()
    ax = fig.gca()

    # ax.scatter(range(N), sample_slice[:, 0])

    ax.hist(sample_naive, range=[-5, 5], histtype='step', density=True, label='naive')
    ax.hist(sample_direct, range=[-5, 5], histtype='step', density=True, label='direct')

    ax.set_xlim([-5, 5])
    
    # ======
    # === 2D
    # ======

    N = 10000
    sample_naive = Naive.sample([N, 2], pdf, xlim=[-5, 5], ylim=np.prod(pdf([0, 0])))
    sample_direct = np.random.normal(size=[N, 2])
    print('sample_naive.shape:', sample_naive.shape)
    print('sample_direct.shape:', sample_direct.shape)

    fig = plt.figure()
    ax = fig.gca()

    hist_naive, xbins, ybins = np.histogram2d(sample_naive[:, 0],
                                              sample_naive[:, 1],
                                              bins=30,
                                              range=[[-5, 5], [-5, 5]],
                                              density=True)
    hist_direct, _, _ = np.histogram2d(sample_direct[:, 0],
                                       sample_direct[:, 1],
                                       bins=[xbins, ybins], density=True)

    print('np.max(hist_naive):', np.max(hist_naive))
    print('np.max(hist_direct):', np.max(hist_direct))

    # Move anchor of `xbins` and `ybins` from upper left to center.
    xcenter = np.diff(xbins)/2 + xbins[:-1]
    ycenter = np.diff(ybins)/2 + ybins[:-1]

    ax.contour(xcenter, ycenter, hist_naive,
               levels=np.asarray([0, 0.1, 0.5, 0.9])*np.max(hist_naive),
               cmap='Blues', label='naive',
               linewidths=2, linestyles='dashed', antialiased=True)
    ax.contour(xcenter, ycenter, hist_direct,
               levels=np.asarray([0, 0.1, 0.5, 0.9])*np.max(hist_direct),
               cmap='Oranges', label='direct',
               linewidths=2, linestyles='dashed', antialiased=True)

    ax.hexbin(sample_naive[:, 0], sample_naive[:, 1], cmap='Blues', edgecolors=None, alpha=0.5)
    # ax.hist2d(sample_naive[:, 0], sample_naive[:, 1],
    #           bins=30, range=[[-5, 5], [-5, 5]],
    #           vmin=0)

    mean_naive = np.sum(sample_naive, axis=1) / N
    mean_direct = np.sum(sample_direct, axis=1) / N

    ax.scatter(mean_naive[0], mean_naive[1], marker='x', color='tab:blue')
    ax.scatter(mean_direct[0], mean_direct[1], marker='x', color='tab:orange')

    # cond = (np.sqrt(np.sum(sample_naive**2 - 0, axis=1)) <= 1)
    # print(cond)
    # ax.scatter(sample_naive[cond][:, 0], sample_naive[cond][:, 1], marker='x', color='tab:blue')
    

    ax.grid()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    # fig, axs = plt.subplots(ncols=2, nrows=2)
    # plot_diff(axs[0][0], sample_naive, sample_ref)
    # plot_diff(axs[1][1], sample_direct, sample_ref)


    plt.show()

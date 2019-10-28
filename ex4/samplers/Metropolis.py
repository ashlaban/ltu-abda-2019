
import time

import numpy as np

from ._base import SamplerBase

class Metropolis(SamplerBase):
    '''
    '''

    def sample(shape, pdf, x0=None, proposal_generator=None,
               dtype=np.float):
        '''Use metropolis sampling to sample from target pdf.

        

        Arguments
        ---------
        shape:
        pdf:

        Keyword Arguments
        -----------------
        x0:
        proposal_generator:
        dtype:

        Returns
        -------
        A vector of shape `shape` sampled from the given pdf.
        '''

        if isinstance(shape, int):
            shape = [shape, 1]

        if proposal_generator is None:
            proposal_generator = np.random.normal

        nsamples = np.prod(shape[:-1])
        ndims = shape[-1]

        samples = np.empty(shape=shape, dtype=dtype)

        if x0 is None:
            x0 = np.zeros(ndims)

        loglikelihood_x0 = np.sum(np.log(pdf(x0)))

        time_start = time.time()
        naccepted = 0
        for iSamp in range(nsamples):
            x1 = x0 + proposal_generator(size=ndims, scale=1.5)

            loglikelihood_x1 = np.sum(np.log(pdf(x1)))

            p_move = min(0, (loglikelihood_x1-loglikelihood_x0))
            y = np.log(np.random.uniform())

            if y <= p_move:
                x0 = x1
                loglikelihood_x0 = loglikelihood_x1
                naccepted += (y <= p_move)

            samples[iSamp, :] = x0
        time_end = time.time()
        total_time = time_end - time_start
        print(f'Efficiency (metropolis): {naccepted/nsamples:.5}')
        print(f'Time taken (metropolis): {total_time:.5}s ({total_time/nsamples:.5}s per sample)')

        return samples.reshape(shape)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import scipy.stats

    pdf = scipy.stats.norm.pdf

    # ======
    # === 1D
    # ======

    N = 10000
    sample_metro = Metropolis.sample([N, 1], pdf)
    sample_direct = np.random.normal(size=[N, 1])
    print('sample_metro.shape:', sample_metro.shape)
    print('sample_direct.shape:', sample_direct.shape)

    fig = plt.figure()
    ax = fig.gca()

    # ax.scatter(range(N), sample_slice[:, 0])

    ax.hist(sample_metro, range=[-5, 5], histtype='step', density=True, label='metro')
    ax.hist(sample_direct, range=[-5, 5], histtype='step', density=True, label='direct')

    ax.set_xlim([-5, 5])
    
    # ======
    # === 2D
    # ======

    N = 10000
    sample_metro = Metropolis.sample([N, 2], pdf)
    sample_direct = np.random.normal(size=[N, 2])
    print('sample_metro.shape:', sample_metro.shape)
    print('sample_direct.shape:', sample_direct.shape)

    fig = plt.figure()
    ax = fig.gca()

    hist_metro, xbins, ybins = np.histogram2d(sample_metro[:, 0],
                                              sample_metro[:, 1],
                                              bins=30,
                                              range=[[-5, 5], [-5, 5]],
                                              density=True)
    hist_direct, _, _ = np.histogram2d(sample_direct[:, 0],
                                       sample_direct[:, 1],
                                       bins=[xbins, ybins], density=True)

    print('np.max(hist_metro):', np.max(hist_metro))
    print('np.max(hist_direct):', np.max(hist_direct))

    # Move anchor of `xbins` and `ybins` from upper left to center.
    xcenter = np.diff(xbins)/2 + xbins[:-1]
    ycenter = np.diff(ybins)/2 + ybins[:-1]

    ax.contour(xcenter, ycenter, hist_metro,
               levels=np.asarray([0, 0.1, 0.5, 0.9])*np.max(hist_metro),
               cmap='Blues', label='metro',
               linewidths=2, linestyles='dashed', antialiased=True)
    ax.contour(xcenter, ycenter, hist_direct,
               levels=np.asarray([0, 0.1, 0.5, 0.9])*np.max(hist_direct),
               cmap='Oranges', label='direct',
               linewidths=2, linestyles='dashed', antialiased=True)

    ax.hexbin(sample_metro[:, 0], sample_metro[:, 1], cmap='Blues', edgecolors=None, alpha=0.5)
    # ax.hist2d(sample_metro[:, 0], sample_metro[:, 1],
    #           bins=30, range=[[-5, 5], [-5, 5]],
    #           vmin=0)

    mean_metro = np.sum(sample_metro, axis=1) / N
    mean_direct = np.sum(sample_direct, axis=1) / N

    ax.scatter(mean_metro[0], mean_metro[1], marker='x', color='tab:blue')
    ax.scatter(mean_direct[0], mean_direct[1], marker='x', color='tab:orange')

    # cond = (np.sqrt(np.sum(sample_metro**2 - 0, axis=1)) <= 1)
    # print(cond)
    # ax.scatter(sample_metro[cond][:, 0], sample_metro[cond][:, 1], marker='x', color='tab:blue')
    

    ax.grid()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    # fig, axs = plt.subplots(ncols=2, nrows=2)
    # plot_diff(axs[0][0], sample_metro, sample_ref)
    # plot_diff(axs[1][1], sample_direct, sample_ref)


    plt.show()

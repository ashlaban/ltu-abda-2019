
import time

import numpy as np

from ._base import SamplerBase

class Slice(SamplerBase):
    '''
    '''

    def sample(shape, pdf, x0=None, burnin=0.1,
               dtype=None, max_iter=100, step_size=10):
        '''Use Slice sampling to sample from target pdf

        Slice sampling generates a markov chain where each subsequent
        pseudo-sample is changed only in a single variable at a time. A
        sample is recorded once all dimensions have been modified. Due to
        this iterative approach, samples are correlated.

        In some sense the procedure is the inverse of rejection sampling where
        one defines a position, and depending on a variate compared to the
        densisty at that point, wither accpets on rejects that point. Slice
        sampling directly generates a point which would be accepted by the
        rejection sample procedure, by first selecting an acceptance level and
        then sampling points from that acceptance level. This is not a full
        proof but might be helpful for intuition.

        Note: Currently this function assumes the output of `pdf` is the
              logarithm of the actual sample distribution.

        Arguments
        ---------
        shape : int, N; or array-like, (N, M, ..., D)
            If int: Treated internally as `[N, 1]`, but the output shape is
            retained as `(N,)`.
            If array-like: Generates `N*M*...` samples of dimensionality `D`.
        pdf : fun(x) -> scalar
            Function to calculate the distribution to sample from. Need not be
            a normalised probability distribution, but should be
            multiplicativley proportional to the distribution you want to
            sample. Should accept input values of dimensionality of
            `shape[-1]`.
        burnin : float
            If between 0 and 1 (exclusive): Sample and discard
            `burnin*np.prod(shape[:-1])` samples before starting the sampling
            proper.
            
            If 1 or larger: Sample `burnin` number of samples and throw them
            away before the sampling proper begins.

        Keyword Arguments
        -----------------
        x0 : array-like, shape (D,)
            Starting point. Should have the same shape as `shape[-1]`. If not
            given, the zero vector will be used.
        dtype : type
            If not given, default to the `dtype` of `x0`. If `x0` also not
            given, the default is `np.float`.
        max_iter : int
            Max number iterations used to generate the slice interval.
        step_size : float
            The step_size of the slice interval finder. Should be adapted to
            match the typical horizontal slice length of the sample
            distribution.

        Returns
        -------
        A vector of shape `shape` sampled from the given pdf.
        '''

        def find_slice(x0, y, pdf, step_size, dim=0, max_iter=100, scheme='linear'):
            '''
            '''
            # TODO: Split into two function, make choice earler in code hierarchy.
            #       Leaf choices are inefficient.
            #  stepsize: must be array-like
            w = step_size[iDim]
            m = max_iter

            x1 = np.empty_like(x0)
            x1[:] = x0

            if scheme == 'linear':
                r = np.random.uniform()

                # Note: Interval assignment must be random for correctness
                L = x0[dim] - r*w
                R = L + w

                # Note: m, j, and k important for correctness
                # TODO: Why?
                j = int(m*np.random.uniform())
                k = int(m-1-j)

                x1[dim] = L
                while (np.sum(pdf(x1)) > y) and (j>0):
                    L -= w
                    j -= 1
                    x1[dim] = L

                x1[dim] = R
                while (np.sum(pdf(x1)) > y) and (k>0):
                    R += w
                    k -= 1
                    x1[dim] = R
                
            elif scheme == 'exponential':
                raise NotImplementedError()

            assert(L < x0[dim] < R)
            return (L, R)

        def _sample(pdf, x0, nsamples, ndims, step_size, dtype):
            '''
            '''

            samples = np.empty(shape=(burnin, ndims), dtype=dtype)

            x1 = np.empty_like(x0)
            x1[:] = x0

            time_start = time.time()

            for iSamp in range(nsamples):
                for iDim in range(ndims):
                    loglikelihood = np.sum(pdf(x0))
                    y = np.log(np.random.uniform()) + loglikelihood
                    L, R = find_slice(x0, y, pdf, dim=iDim, step_size=step_size)

                    x1[iDim] = np.random.uniform(L, R)
                    while np.sum(pdf(x1)) < y:
                        # Shrinkage
                        if x1[iDim] < x0[iDim]:
                            L = float(x1[iDim])
                        else:
                            R = float(x1[iDim])
                        x1[iDim] = np.random.uniform(L, R)
                    x0[iDim] = x1[iDim]

                samples[iSamp, :] = x1

            time_end = time.time()
            total_time = time_end - time_start

            return samples, x0, total_time

        # ============================
        # === Function 'sample' proper
        # ============================

        orig_shape = shape
        if isinstance(shape, int):
            shape = [shape, 1]

        if isinstance(step_size, int):
            step_size = [step_size, 1]

        nsamples = np.prod(shape[:-1])
        ndims = shape[-1]

        if burnin < 1:
            burnin = burnin * nsamples

        if x0 is None:
            x0 = np.zeros(shape[1:], dtype=np.float)

        if dtype is None:
            dtype = x0.dtype

        if burnin:
            _, x0, _ = _sample(pdf, x0, burnin, ndims,
                               step_size=step_size, dtype=dtype)
        samples, _, total_time = _sample(pdf, x0, nsamples, ndims,
                                         step_size=step_size, dtype=dtype)

        print(f'Efficiency (slice): 1.0')
        print(f'Time taken (slice): {total_time:.5}s ({total_time/nsamples:.5}s per sample)')

        return samples.reshape(orig_shape)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import scipy.stats

    pdf = lambda x: np.log(scipy.stats.norm.pdf(x))

    # ======
    # === 1D
    # ======

    N = 1000
    sample_slice = Slice.sample([N, 1], pdf)
    sample_direct = np.random.normal(size=[N, 1])
    print('sample_slice.shape:', sample_slice.shape)
    print('sample_direct.shape:', sample_direct.shape)

    fig = plt.figure()
    ax = fig.gca()

    # ax.scatter(range(N), sample_slice[:, 0])

    ax.hist(sample_slice, range=[-5, 5], histtype='step', density=True, label='metro')
    ax.hist(sample_direct, range=[-5, 5], histtype='step', density=True, label='direct')

    ax.set_xlim([-5, 5])
    
    # ======
    # === 2D
    # ======

    N = 1000
    sample_slice = Slice.sample([N, 2], pdf)
    sample_direct = np.random.normal(size=[N, 2])
    print('sample_slice.shape:', sample_slice.shape)
    print('sample_direct.shape:', sample_direct.shape)

    fig = plt.figure()
    ax = fig.gca()

    hist_slice, xbins, ybins = np.histogram2d(sample_slice[:, 0],
                                              sample_slice[:, 1],
                                              bins=30,
                                              range=[[-5, 5], [-5, 5]],
                                              density=True)
    hist_direct, _, _ = np.histogram2d(sample_direct[:, 0],
                                       sample_direct[:, 1],
                                       bins=[xbins, ybins], density=True)

    print('np.max(hist_slice):', np.max(hist_slice))
    print('np.max(hist_direct):', np.max(hist_direct))

    # Move anchor of `xbins` and `ybins` from upper left to center.
    xcenter = np.diff(xbins)/2 + xbins[:-1]
    ycenter = np.diff(ybins)/2 + ybins[:-1]

    ax.contour(xcenter, ycenter, hist_slice,
               levels=np.asarray([0, 0.1, 0.5, 0.9])*np.max(hist_slice),
               cmap='Blues', label='metro',
               linewidths=2, linestyles='dashed', antialiased=True)
    ax.contour(xcenter, ycenter, hist_direct,
               levels=np.asarray([0, 0.1, 0.5, 0.9])*np.max(hist_direct),
               cmap='Oranges', label='direct',
               linewidths=2, linestyles='dashed', antialiased=True)

    ax.hexbin(sample_slice[:, 0], sample_slice[:, 1], cmap='Blues', edgecolors=None, alpha=0.5)
    # ax.hist2d(sample_slice[:, 0], sample_slice[:, 1],
    #           bins=30, range=[[-5, 5], [-5, 5]],
    #           vmin=0)

    mean_slice = np.sum(sample_slice, axis=1) / N
    mean_direct = np.sum(sample_direct, axis=1) / N

    ax.scatter(mean_slice[0], mean_slice[1], marker='x', color='tab:blue')
    ax.scatter(mean_direct[0], mean_direct[1], marker='x', color='tab:orange')

    # cond = (np.sqrt(np.sum(sample_slice**2 - 0, axis=1)) <= 1)
    # print(cond)
    # ax.scatter(sample_slice[cond][:, 0], sample_slice[cond][:, 1], marker='x', color='tab:blue')
    

    ax.grid()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    # fig, axs = plt.subplots(ncols=2, nrows=2)
    # plot_diff(axs[0][0], sample_slice, sample_ref)
    # plot_diff(axs[1][1], sample_direct, sample_ref)


    plt.show()

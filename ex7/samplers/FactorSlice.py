
import time
import threading
import multiprocessing

import numpy as np
import tqdm

import numba
from numba import jit, vectorize, objmode

from ._base import SamplerBase
from .Slice import Slice

class FactorSlice(SamplerBase):
    '''
    '''

    def sample(shape, pdf, x0=None, burnin=0.1,
               dtype=None, max_iter=100, step_size=10,
               njobs=1, show_progress=True):
        '''Use Factor Slice sampling to sample from target pdf

        Slice sampling generates a markov chain where each subsequent
        pseudo-sample is changed only in a single variable at a time. A
        sample is recorded once all dimensions have been modified. Due to
        this iterative approach, samples are correlated.

        Factor slicing is very similar, but instead of stepping in an
        axis-aligned direction at each step, the coordianate system is rotated
        to minimise correlation.

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
        show_progress : bool
            If True, shows progress.

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

        @jit(nopython=True, nogil=True)
        def find_slice(x0, y, pdf, W, E, dim=0, max_iter=200):
            '''
            '''
            w = W[dim]*E[:, dim]
            # w = np.real(E[dim, :]).astype(x0.dtype)
            m = max_iter

            L = x0 - np.random.uniform(0, 1)*w
            R = L + w

            # Note: m, j, and k important for correctness
            # TODO: Why?
            j = int(m*np.random.uniform(0, 1))
            k = int(m-1-j)

            while (pdf(L) > y) and (j>0):
                L -= w
                j -= 1

            while (pdf(R) > y) and (k>0):
                R += w
                k -= 1

            # assert((L <= x0).all() and (x0 <= R).all())
            return (L, R)

        @jit(nopython=True, nogil=True)
        def _sample_inplace(pdf, x0, dst, W, E, start=0, stop=-1, ndims=-1,
                            show_progress=False, show_progress_leave=False,
                            show_progress_prefix='Sample',
                            show_progress_offset=0):
            '''
            '''

            if stop == -1:
                stop = len(dst)

            if ndims == -1:
                ndims = len(x0)

            x1 = np.empty_like(x0)
            x1[:] = x0

            if show_progress_offset > 0:
                for i in range(show_progress_offset):
                    print()
                for i in range(show_progress_offset):
                    print('\u001b[2A')

            # time_start = time.time()

            for iSamp in range(start, stop):
                for iDim in np.random.permutation(np.arange(ndims)):
                    loglikelihood = pdf(x0)
                    y = np.log(np.random.uniform(0, 1)) + loglikelihood
                    L, R = find_slice(x0, y, pdf, W, E, dim=iDim)
                    x1 = (L + np.random.uniform(0, 1)*(R-L)).astype(x0.dtype)

                    grand_idx = np.argmax(E[:, iDim])

                    # print('Dist: ', np.linalg.norm(x1-x0))

                    dummy = 0
                    while pdf(x1) < y:
                        dummy += 1

                        # Shrinkage
                        A = np.sign(L - x0)
                        B = np.sign(R - x0)
                        C = np.sign(x1 - x0)

                        idx = np.ones_like(A, dtype=np.bool_)
                        idx = np.logical_and(idx, A!=0)
                        idx = np.logical_and(idx, B!=0)
                        idx = np.logical_and(idx, C!=0)

                        A = A[idx]
                        B = B[idx]
                        C = C[idx]

                        if (A != -1*B).all():
                            break
                            print('Dist: ', dummy, '', np.linalg.norm(x1-x0))
                            idx_ = np.where((A != -1*B))
                            print(A)
                            print(B)
                            print(C[grand_idx])
                            print()
                            print()
                            print(L[idx_])
                            print(R[idx_])
                            print(x0[idx_])
                            print()
                            print()
                            print(E[:, iDim][idx_])
                            print(W)
                            assert(False)

                        # if dummy > 1:
                        #     print('Dist: ', dummy, '', np.linalg.norm(x1-x0), a, b, c)
                        #     print(np.sign(L - x0))
                        #     print(np.sign(R - x0))

                        # if dummy > 1:
                            # print(x1[:5])
                            # print(x0[:5])

                        # if (np.sign(R - x0) == np.sign(x1 - x0)).all():
                        if (B == C).all():
                            R[:] = x1
                        else:
                            L[:] = x1
                        x1 = (L + np.random.uniform(0.01, 0.99)*(R-L)).astype(x0.dtype)
                        if dummy == 100:
                            x1[:] = x0
                    x0[:] = x1
                dst[iSamp, :] = x1

                if ((iSamp % (100)) == 0) and show_progress:
                    for i in range(show_progress_offset):
                        print('\u001b[1B\u001b[1A')
                    print('\u001b[2K\u001b[1A')  # Clear entire line and move up 1
                    print('\u001b[2K', show_progress_prefix, iSamp, '/', stop, '\u001b[1A')
                    for i in range(show_progress_offset):
                        print('\u001b[2A')

            if show_progress:
                for i in range(show_progress_offset):
                    print('\u001b[1B\u001b[1A')

                if show_progress_leave:
                    print(show_progress_prefix, iSamp+1, '/', stop, '\u001b[1A')
                else:
                    print('\u001b[2K\u001b[1A')

                for i in range(show_progress_offset):
                    print('\u001b[2A')

            # time_end = time.time()
            # total_time = time_end - time_start
            # print(total_time)

            # print('x0.shape: ', x0.shape)

            return dst, x0#, total_time

        @jit(nopython=True, nogil=True)
        def _sample(pdf, x0, nsamples, ndims, W, E, dtype,
                    show_progress=False, show_progress_leave=False,
                    show_progress_prefix='Sample'):
            samples = np.empty(shape=(nsamples, ndims), dtype=dtype)
            return _sample_inplace(pdf, x0, dst=samples, W=W, E=E,
                                   start=0, stop=nsamples,
                                   show_progress=show_progress,
                                   show_progress_leave=show_progress_leave,
                                   show_progress_prefix=show_progress_prefix)
            

        # ============================
        # === Function 'sample' proper
        # ============================

        start_time = time.time()

        orig_shape = shape
        if isinstance(shape, int):
            shape = [shape, 1]
        # if isinstance(shape, np.ndarray) and len(shape.shape) == 1:
        #     shape = [shape[0], 1]

        nsamples = int(np.prod(shape[:-1]))
        ndims = int(shape[-1])

        if isinstance(step_size, int) or isinstance(step_size, float):
            step_size = np.ones(shape=(ndims,))*step_size

        if burnin < 1:
            burnin = burnin * nsamples
        burnin = max(100, int(burnin))

        if x0 is None:
            if dtype is None:
                x0 = np.zeros(shape[1:], dtype=np.float)
            else:
                x0 = np.zeros(shape[1:], dtype=dtype)

        if dtype is None:
            dtype = x0.dtype

        tuning_shape = orig_shape[:]
        tuning_shape[0] = burnin
        assert(tuning_shape != orig_shape)

        tuning_sample = Slice.sample(tuning_shape, pdf,
                                     x0=x0, burnin=False, dtype=dtype,
                                     max_iter=max_iter, step_size=step_size,
                                     show_progress=True)

        x0 = tuning_sample[-1, :]
        step_size = np.std(tuning_sample, axis=0)

        tuning_sample = Slice.sample(tuning_shape, pdf,
                                     x0=x0, burnin=False, dtype=dtype,
                                     max_iter=max_iter, step_size=step_size,
                                     show_progress=True)
        x0 = tuning_sample[-1, :]

        cov = np.cov(tuning_sample, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        W = np.real(np.sqrt(np.abs(eigenvals))).astype(dtype)
        E = np.real(eigenvecs).astype(dtype)

        # np.save('tuning_sample', tuning_sample)
        # np.save('cov', cov)
        # import sys
        # sys.exit(0)

        if njobs < 2:
            samples, _ = _sample(pdf, x0, nsamples, ndims,
                                 W=W, E=E, dtype=dtype,
                                 show_progress=True, show_progress_leave=True,
                                 show_progress_prefix='Sample')
        else:
            samples = np.empty(shape=(nsamples, ndims), dtype=dtype)

            jobs = []
            nsamples_remaining = nsamples
            chunk_size = int(nsamples/njobs)
            for iJob in range(njobs):
                current_chunk_size = min(chunk_size, nsamples_remaining)
                chunk_start = iJob * chunk_size
                chunk_stop = chunk_start + current_chunk_size
                nsamples_remaining -= current_chunk_size

                # print('Job {}: Indices {}, {}'.format(iJob,
                #                                       chunk_start,
                #                                       chunk_stop))

                jobs.append(threading.Thread(target=_sample_inplace,
                                             args=(pdf, x0, samples, W, E),
                                             kwargs=dict(start=chunk_start,
                                                         stop=chunk_stop,
                                                         show_progress=True,
                                                         show_progress_leave=True,
                                                         show_progress_offset=iJob,
                                                         show_progress_prefix='Job {}:'.format(iJob))))
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()

        # print(f'Efficiency (slice): 1.0')
        # print(f'Time taken (slice): {total_time:.5}s ({total_time/nsamples:.5}s per sample)')

        end_time = time.time()
        total_time = end_time - start_time

        if show_progress:
            print('\u001b['+str(njobs)+'B')
            print(f'Time taken: {total_time:.3f} seconds')

        return samples.reshape(orig_shape)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import scipy.stats

    pdf = lambda x: np.log(scipy.stats.norm.pdf(x))

    # ======
    # === 1D
    # ======

    # N = 1000
    # sample_slice = FactorSlice.sample([N, 1], pdf)
    # sample_direct = np.random.normal(size=[N, 1])
    # print('sample_slice.shape:', sample_slice.shape)
    # print('sample_direct.shape:', sample_direct.shape)

    # fig = plt.figure()
    # ax = fig.gca()

    # # ax.scatter(range(N), sample_slice[:, 0])

    # ax.hist(sample_slice, range=[-5, 5], histtype='step', density=True, label='metro')
    # ax.hist(sample_direct, range=[-5, 5], histtype='step', density=True, label='direct')

    # ax.set_xlim([-5, 5])
    
    # ======
    # === 2D
    # ======

    N = 1000
    sample_slice = FactorSlice.sample([N, 2], pdf)
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

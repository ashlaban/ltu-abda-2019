
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import util

from data import raw_y, ind, ids, cnts, is_child, is_child_row, x1
from model import gen_sampler_pdf
from samplers import Slice as Slice

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', type=int, default=0,
                    help='')
parser.add_argument('-n', type=int, default=1000,
                    help='')

parser.add_argument('-s', '--step-size', type=float, default=10.,
                    help='')

parser.add_argument('-j', '--num-jobs', type=int, default=1,
                    help='WARN: Problem with the GIL and numba, will probably deadlock if num_jobs > 1.')

parser.add_argument('--profile', action='store_true',
                    help='')

# parser.add_argument('--ex5', action='store_true',
#                     help='')

parser.add_argument('--standardise', action='store_true',
                    help='')

parser.add_argument('--no-save', action='store_true',
                    help='')

parser.add_argument('--report', action='store_true',
                    help='')

parser.add_argument('--do-special-trimming', action='store_true',
                    help='Don\'t use this!')

args = parser.parse_args()

id = args.id
N = args.n

L = len(ids)
F = 7+2*L

# if args.ex5:
#     is_child_row = np.zeros_like(is_child_row)

log_y = np.log(raw_y)
log_y_mean = log_y.mean()
log_y_std = log_y.std()

if args.standardise:
    z = (log_y - log_y_mean) / log_y_std
    zx1 = (x1 - x1.mean()) / x1.std()
else:
    z = log_y
    zx1 = x1

x0 = np.concatenate([np.ones(shape=L, dtype=np.float32)*5,
                     np.ones(shape=L, dtype=np.float32)*1,
                     np.asarray([1, 5, 5, 1, 1, 1, 1],
                    dtype=np.float32)])
pdf = gen_sampler_pdf(z=z, n_theta=L,
                      id=ind, x0=is_child_row, x1=zx1)

if args.profile:
    import cProfile
    cProfile.run('Slice.sample([N, F], pdf, x0=x0, step_size=args.step_size)', 'profile')
else:
    sample_slice = Slice.sample([N, F], pdf, x0=x0, step_size=args.step_size, njobs=args.num_jobs)

eta0 = sample_slice[:, 0:L]
eta1 = sample_slice[:, L:2*L]
sample_slice[:, 0:L] = (sample_slice[:, -6].reshape(-1, 1) +
                        sample_slice[:, -2].reshape(-1, 1)*is_child_row.reshape(1, -1) +
                        sample_slice[:, -4].reshape(-1, 1)*eta0)
sample_slice[:, L:2*L] = (sample_slice[:, -5].reshape(-1, 1) +
                          sample_slice[:, -1].reshape(-1, 1)*is_child_row.reshape(1, -1) +
                          sample_slice[:, -3].reshape(-1, 1)*eta1)

if args.standardise:
    sample_slice[:, 0:L] *= log_y_std
    sample_slice[:, 0:L] += log_y_mean

    sample_slice[:, L:2*L] *= log_y_std
    sample_slice[:, L:2*L] *= x1.std()

    sample_slice[:, -7] *= log_y_std # sigma

    sample_slice[:, -6] *= log_y_std  # mu0
    sample_slice[:, -6] += log_y_mean # mu0

    sample_slice[:, -5] *= log_y_std  # mu1
    # sample_slice[:, -5] += log_y_mean # mu1

    sample_slice[:, -4] *= log_y_std # tau0
    sample_slice[:, -3] *= log_y_std # tau1

    sample_slice[:, -2] *= log_y_std # phi0
    sample_slice[:, -1] *= log_y_std # phi1

if args.do_special_trimming:
    # ess = int(util.ess(sample_slice))

    idx = np.random.choice(N, 100)
    sample_slice = sample_slice[idx, :]

    # Hmm, a random subsample seems to have _way_ worse ESS than the full
    # sample. Could it be that the sample is still correlated at lengths
    # comparable to N?
    # 
    # Next question: How to find the maximally uncorrelated subsample? (Without
    # testing all ^^)
    #  - Greedy approach, remove the sample contributing the most to
    #    autocorrelation across all timesteps. Repeat until target size is
    #    raught (this is (old) english btw).
    #  - For extracting a small subsample from a very large one the random
    #    approch should be better?

if args.report:
    ess = util.ess(sample_slice).astype(int)

    string = str(sample_slice[:, :L].mean(axis=0)).replace('\n', '\n             ')
    print(f'Mean theta0: {string}')
    string = str(sample_slice[:, L:2*L].mean(axis=0)).replace('\n', '\n             ')
    print(f'     theta1: {string}')
    print(f'     sigma : {sample_slice[:, -7].mean(axis=0)}')
    print(f'     mu0   : {sample_slice[:, -6].mean(axis=0)}')
    print(f'     mu1   : {sample_slice[:, -5].mean(axis=0)}')
    print(f'     tau0  : {sample_slice[:, -4].mean(axis=0)}')
    print(f'     tau1  : {sample_slice[:, -3].mean(axis=0)}')
    print(f'     phi0  : {sample_slice[:, -2].mean(axis=0)}')
    print(f'     phi1  : {sample_slice[:, -1].mean(axis=0)}')
    print()

    string = str(ess[:L]).replace('\n', '\n             ')
    print(f'ESS  theta0: {string}')
    string = str(ess[L:2*L]).replace('\n', '\n             ')
    print(f'     theta1: {string}')
    print(f'     sigma : {ess[-7]}')
    print(f'     mu0   : {ess[-6]}')
    print(f'     mu1   : {ess[-5]}')
    print(f'     tau0  : {ess[-4]}')
    print(f'     tau1  : {ess[-3]}')
    print(f'     phi0  : {ess[-2]}')
    print(f'     phi1  : {ess[-1]}')

# if args.ex5:
#     # In exercise 5 we do not use the is_child variable (always zero) hence it
#     # can be removed.
#     sample_slice = np.delete(sample_slice, -1, axis=1)

if not args.profile and not args.no_save:
    ex5_ext = ''
    np.savez(f'sample{ex5_ext}-{N}-{id}.npz', data=sample_slice)

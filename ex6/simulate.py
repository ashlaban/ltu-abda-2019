
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import util

from data import raw_y, ind, ids, cnts, is_child_row
from model import gen_sampler_pdf
from samplers import Slice

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', type=int, default=0,
                    help='')
parser.add_argument('-n', type=int, default=1000,
                    help='')

parser.add_argument('-s', '--step-size', type=float, default=10.,
                    help='')

parser.add_argument('-j', '--num-jobs', type=int, default=1,
                    help='')

parser.add_argument('--profile', action='store_true',
                    help='')

parser.add_argument('--ex5', action='store_true',
                    help='')

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
F = 4+len(ids)

if args.ex5:
    is_child_row = np.zeros_like(is_child_row)

log_y = np.log(raw_y)
log_y_mean = log_y.mean()
log_y_std = log_y.std()

if args.standardise:
    z = (log_y - log_y_mean) / log_y_std
else:
    z = log_y

x0 = np.concatenate([np.ones(shape=len(ids), dtype=np.float32)*5, np.asarray([1, 5, 1, 1], dtype=np.float32)])
pdf = gen_sampler_pdf(z=z, id=ind, child_id=is_child_row, n_theta=len(ids))

if args.profile:
    import cProfile
    cProfile.run('Slice.sample([N, F], pdf, x0=x0, step_size=args.step_size)', 'profile')
else:
    sample_slice = Slice.sample([N, F], pdf, x0=x0, step_size=args.step_size, njobs=args.num_jobs)

if args.standardise:
    sample_slice[:, :len(ids)] *= log_y_std
    sample_slice[:, :len(ids)] += log_y_mean

    sample_slice[:, -3] *= log_y_std # mu
    sample_slice[:, -3] += log_y_mean # mu
    sample_slice[:, -1] *= log_y_std # phi

    sample_slice[:, -4] *= log_y_std # sigma
    sample_slice[:, -2] *= log_y_std # tau

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
    ess = util.ess(sample_slice)
    string = str(sample_slice[:, :len(ids)].mean(axis=0)).replace('\n', '\n            ')
    print(f'Mean theta: {string}')
    print(f'     sigma: {sample_slice[:, -4].mean(axis=0)}')
    print(f'     mu   : {sample_slice[:, -3].mean(axis=0)}')
    print(f'     phi  : {sample_slice[:, -1].mean(axis=0)}')
    print(f'     tau  : {sample_slice[:, -2].mean(axis=0)}')
    print()

    string = str(ess[:len(ids)]).replace('\n', '\n            ')
    print(f'ESS  theta: {string}')
    print(f'     sigma: {ess[-4]}')
    print(f'     mu   : {ess[-3]}')
    print(f'     phi  : {ess[-1]}')
    print(f'     tau  : {ess[-2]}')

if args.ex5:
    # In exercise 5 we do not use the is_child variable (always zero) hence it
    # can be removed.
    sample_slice = np.delete(sample_slice, -1, axis=1)

if not args.profile and not args.no_save:
    ex5_ext = '-ex5' if args.ex5 else ''
    np.savez(f'sample{ex5_ext}-{N}-{id}.npz', data=sample_slice)

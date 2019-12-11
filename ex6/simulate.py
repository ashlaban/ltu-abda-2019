
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

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
args = parser.parse_args()

id = args.id
N = args.n
F = 4+len(ids)

if args.ex5:
    is_child_row = np.zeros_like(is_child_row)

x0 = np.concatenate([np.ones(shape=len(ids), dtype=np.float32)*5, np.asarray([1, 5, 1, 1], dtype=np.float32)])
pdf = gen_sampler_pdf(z=np.log(raw_y), id=ind, child_id=is_child_row, n_theta=len(ids))

if args.profile:
    import cProfile
    cProfile.run('Slice.sample([N, F], pdf, x0=x0, step_size=args.step_size)', 'profile')
else:
    sample_slice = Slice.sample([N, F], pdf, x0=x0, step_size=args.step_size, njobs=args.num_jobs)

if args.ex5:
    sample_slice = np.delete(sample_slice, -1, axis=1)

if not args.profile:
    ex5_ext = '-ex5' if args.ex5 else ''
    np.savez(f'sample{ex5_ext}-{N}-{id}.npz', data=sample_slice)

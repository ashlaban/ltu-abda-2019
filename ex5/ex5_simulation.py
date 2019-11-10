
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from ex5_data import raw_y, ind, ids, cnts
from ex5_model import gen_sampler_pdf
from samplers import Slice

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', type=int, default=0,
                    help='')
parser.add_argument('-n', type=int, default=1000,
                    help='')
args = parser.parse_args()

id = args.id
N = args.n
F = 3+len(ids)

x0 = np.concatenate([np.ones(shape=len(ids))*5, np.asarray([1, 5, 1])])
pdf = gen_sampler_pdf(z=np.log(raw_y), id=ind, n_theta=len(ids))
sample_slice = Slice.sample([N, F], pdf, x0=x0, step_size=10)

np.savez(f'sample-{N}-{id}.npz', data=sample_slice)

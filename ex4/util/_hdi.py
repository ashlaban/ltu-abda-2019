
import numpy as np

def hdi(sample, cred_mass=0.95, *args, method='std', **kwargs):
    assert(method in ['std', 'hist'])
    if method == 'std':
        return hdi_std(sample, cred_mass,*args, **kwargs)
    elif method == 'hist':
        h, x = np.histogram(sample, density=True, bins=kwargs['bins'])
        return hdi_hist(h, x, *args, **kwargs)

def hdi_std(sample, cred_mass=0.95):
    x = sorted(sample)
    ciIdxInc = int(np.ceil(cred_mass*len(sample)))
    nCIs = len(sample) - ciIdxInc
    ciWidth = np.zeros(nCIs)
    for i in range(nCIs):
        ciWidth[i] = x[i+ciIdxInc] - x[i]

    hdi_min = x[np.argmin(ciWidth)]
    hdi_max = x[np.argmax(ciWidth) + ciIdxInc]

    return (hdi_min, hdi_max)


def hdi_hist(h, x, cred_mass=0.95):
    # convert to probability mass
    # h = h*(x[1]-x[0])/np.sum(h)
    h = h/np.sum(h)

    integrated_mass = 0
    bins = []
    while integrated_mass <= cred_mass:
        i = np.argmax(h)
        integrated_mass += h[i]
        bins += [i]
        h[i] = 0

    limits = []
    current_limit = [0, 0]
    bins = sorted(bins)
    for i, iBin in enumerate(bins[:-1]):
        if i == 0:
            current_limit[0] = x[bins[i]]
            continue

        if int(bins[i-1]) == int(bins[i]-1):
            current_limit[1] = x[bins[i]+1]
        else:
            limits += [current_limit]
            current_limit = [0, 0]
    limits += [current_limit]

    if len(limits) == 1:
        return limits[0]
    return limits


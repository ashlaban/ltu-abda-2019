
import numpy as np

def hdi(sample, cred_mass=0.95, *args, method='std', **kwargs):
    '''Find an estimate of the highest density interval

    Arguments
    ---------
    sample : array-like
        Samples from the distribution to approximate hdi for. Should be one
        dimensional due to technical limitations.
    cred_mass : float
        Proportion of probability mass that should be included in HDI.
        Determines length of resulting interval.

    Keyword arguments
    -----------------
    method : str, either ['std' or 'hist']
        The method used to calculate the hdi.

        When `method=='std'`: Uses the implementation of J. K. Kruschke in
        "Doing Bayesian Data Analysis" (Section 25.2.4).

        When `method=='hist'`: Uses a histogram method to iteratively find the
        highest density bins and appoximate the interval(s) through bin edges.
        Theoretically supports multiple disconnected intervals, but this is
        untested.

    Returns
    -------
    tuple (hdi_min, hdi_max)
    '''

    assert(method in ['std', 'hist'])
    if method == 'std':
        return hdi_std(sample, cred_mass,*args, **kwargs)
    elif method == 'hist':
        h, x = np.histogram(sample, density=True, bins=kwargs['bins'])
        return hdi_hist(h, x, *args, **kwargs)

def hdi_std(sample, cred_mass=0.95):
    '''Find an estimate of the highest density interval

    Estimates the highest density interval (HDI) of a given sample by relying
    on the equivalence of the HDI and the shortest contiguous interval
    containing a given mass. This holds only for unimodal distributions.

    The idea is to check each contiguous groping of the sought after number of
    samples and find the grouping that spans the shortest interval

    This can be done in linear time for univariate distributions if the samples
    are sorted (yielding n log n overall).

    Arguments
    ---------
    sample : array-like
        Samples from the distribution to approximate hdi for. Should be one
        dimensional due to technical limitations.
    cred_mass : float
        Proportion of probability mass that should be included in HDI.
        Determines length of resulting interval.

    Returns
    -------
    tuple (hdi_min, hdi_max)
    '''

    x = sorted(sample)

    # Length of shortest interval (in samples) containing given mass.
    ciIdxInc = int(np.ceil(cred_mass*len(sample)))

    # Calculate the resulting spans of the groups
    nCIs = len(sample) - ciIdxInc
    ciWidth = np.zeros(nCIs)
    for i in range(nCIs):
        ciWidth[i] = x[i+ciIdxInc] - x[i]

    # HDI is the shortest interval
    hdi_min = x[np.argmin(ciWidth)]
    hdi_max = x[np.argmax(ciWidth) + ciIdxInc]

    return (hdi_min, hdi_max)


def hdi_hist(h, x, cred_mass=0.95):
    '''Find an estimate of the highest density interval

    Estimates the highest density interval (HDI) of a given sample by
    iteratively finding the histogram bin with the highest density util the
    sought after probability mass is found.

    Technically supports non-unimodal distributions, but this is untested.

    Note: Use ´util.hdi(sample, cred_mass, method='hist')` for an interface
    that accepts a sample and not a pre-binned histogram.

    Arguments
    ---------
    h : array-like, shape (N,)
        A histogram of a sample, such as the one produced by
        `h, x = np.histogram(sample)`.
    x : array-like, shape (N+1,)
        The bin edges corresponding to `h`, such as those produced by
        `h, x = np.histogram(sample)`.

    Returns
    -------
    tuple (hdi_min, hdi_max), or
    list [(hdi_min0, hdi_max0), (hdi_min1, hdi_max1), ...]
        If the procedure returns only a single interval this is output.
        Otherwise a list of all intervals are output.
    '''

    # Convert to probability mass
    h = h/np.sum(h)

    # Find the highest mass bins
    integrated_mass = 0
    bins = []
    while integrated_mass <= cred_mass:
        i = np.argmax(h)
        integrated_mass += h[i]
        bins += [i]
        h[i] = 0

    # Merge contiguous bins to a single interval
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


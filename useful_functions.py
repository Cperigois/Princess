import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def newbin_lin(histobin):
    size = len(histobin)
    delta = (histobin[1]-histobin[0])/2.
    rightbin = histobin+delta
    return rightbin[:(size-1)]
import numpy as np


def m1_m2_to_mc_q(m1, m2):
    """This function does the mapping (m1,m2) --> (mc,q)

    Parameters
    ----------
    m1 : float or numpy array
        Mass of primary of the source(s)
    m2 : float or numpy array
        Mass of seconday of the source(s)

    Returns
    -------
    mc : float or numpy array
        Chirp mass of the sources(s)
    q : float or numpy array
        Mass ratio of the source(s)
    """

    mc = np.power((m1*m2), 0.6) / (np.power(m1 + m2, 0.2))
    q = np.minimum(m2, m1) / np.maximum(m1,m2)

    return mc, q


def mc_q_to_m1_m2(mc, q):
    """This function does the mapping (mc,q) --> (m1,m2)

    Parameters
    ----------
    mc : float or numpy array
        Chirp mass of the sources(s)
    q : float or numpy array
        Mass ratio of the source(s)

    Returns
    -------
    m1 : float or numpy array
        Mass of primary of the source(s)
    m2 : float or numpy array
        Mass of seconday of the source(s)
    """

    m1 = mc*np.power((1.0+q)/(q*q*q), 0.2)
    m2 = q*m1

    return m1, m2

def mt_q_to_m1_m2(Mt,q) :
	m1 = Mt/(1+q)
	m2 = m1*q

	return m1,m2
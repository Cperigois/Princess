import math
import numpy as np
from scipy.integrate import quad

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

def mass1(Mc,Q) :
	return Mc*pow(1+Q,1./5.)*pow(Q,2./5.)

def mass2(Mc,Q) :
	return Mc*pow(1+Q,1./5.)*pow(Q,-3./5.)

def m2_tania(m1,Q):
	return m1*Q

def E(x):
	return math.pow(0.3153*math.pow(1.+float(x),3.)+0.6847,0.5)

def rz(x) :
	return 1./(math.pow(0.3153*math.pow(1.+float(x),3.)+0.6847,0.5))

def dist_lum(x) :
	H0 = 2.183e-18 #s-1
	c = 2.99e10 #cm s-1
	tmp = quad(rz,0,x)
	return c/(H0*(1+x))*tmp[0]

def tau_Myr(x) :
	return(1./(2.183e-18*3600*24*365*1.e6*(1+x)*math.pow(0.3153*math.pow(1.+x,3.)+0.6847,0.5)))

def fe2(e,*args) :
	return (9.99986e-5*math.pow(args[0],1./2.)*math.pow(args[1]*(1.-args[2]*args[2])/(1.-e*e)*math.pow(e/args[2],12./19.)*math.pow((1.+121.*e*e/304.)/(1+121*args[2]*args[2]/304.),870./2299.),-3./2.)-args[3])

def fe3(e,*args) :
	#print(args)
	a = 9.99986e-5*math.pow(args[0],1./2.)*math.pow(args[1]*(1.-args[2]*args[2])/(1.-e*e)*math.pow(e/args[2],12./19.)*math.pow((1.+121.*e*e/304.)/(1+121*args[2]*args[2]/304.),870./2299.),-3./2.)
	b = args[3]
	if (a-b)<0 :
		return(0)
	else :
		return(a-b)


def mass() :
	liste_in = "/home/perigois/These/Astro/Data/Cat/Cat_stochastic_iso_BBHs.dat"
	with open(liste_in, "r") as liste:
		for line in liste:
			Mc,Q,Z_merg,Z_form,time_del,Z,a0,ecc,idd = line.split()
			m1 = mass1(float(Mc),float(Q))
			m2 = mass2(float(Mc),float(Q))
			m2_t = m2_tania(float(m2),float(Q))
			print(Q,' ',m1/m2)

def Xsi(m1,m2,idd) :
	sigmaSpin = 0.1
	v1_L=np.random.normal(0.0,sigmaSpin)
	v2_L=np.random.normal(0.0,sigmaSpin)
	v3_L=np.random.normal(0.0,sigmaSpin)
	V_1 = np.sqrt(v1_L*v1_L+v2_L*v2_L+v3_L*v3_L)

	v1_L=np.random.normal(0.0,sigmaSpin)
	v2_L=np.random.normal(0.0,sigmaSpin)
	v3_L=np.random.normal(0.0,sigmaSpin)
	V_2 = np.sqrt(v1_L*v1_L+v2_L*v2_L+v3_L*v3_L)

	costheta1=2.*np.random.uniform(0.0,1.0)-1.0
	costheta2=2.*np.random.uniform(0.0,1.0)-1.0

	if idd=='dyn':
		return((V_1*costheta1*m1+V_2*costheta2*m2)/(m1+m2))
	else:
		return((V_1*m1+V_2*m2)/(m1+m2))

def Xsi_compil(m1,m2, chi1, chi2) :
	return((m1*np.cos(theta1)*chi1 + m2*np.cos(theta2)*chi2)/(m1+m2))

def fmerg_f(m1,m2,xsi,zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm)
	eta = m1*m2/pow(m1+m2,2.)
	fmerg_mu0 = 1.-4.455*pow(1-xsi,0.217)+3.521*pow(1.-xsi,0.26)
	fmerg_y = 0.6437*eta -0.05822*eta*eta -7.092*eta*eta*eta +0.827*eta*xsi -0.2706*eta*xsi*xsi -3.935*eta*eta*xsi
	return (fmerg_mu0+fmerg_y)/(math.pi*mtot)

def fring_f(m1,m2,xsi,zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm)
	eta = m1*m2/pow(m1+m2,2.)
	fring_mu0 = (1.-0.63*pow(1.-xsi,0.3))/2.
	fring_y = 0.1469*eta -0.0249*eta*eta +2.325*eta*eta*eta - 0.1228*eta*xsi -0.02609*eta*xsi*xsi +0.1701*eta*eta*xsi
	return (fring_mu0+fring_y)/(math.pi*mtot)

def sigma_f(m1,m2,xsi,zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm)
	eta = m1*m2/pow(m1+m2,2.)
	sigma_mu0 =(1.-0.63*pow(1.-xsi,0.3))*pow(1.-xsi,0.45)/4.
	sigma_y = -0.4098*eta +1.829*eta*eta -2.87*eta*eta*eta - 0.03523*eta*xsi +0.1008*eta*xsi*xsi -0.02017*eta*eta*xsi
	return (sigma_mu0+sigma_y)/(math.pi*mtot)

def fcut_f(m1,m2,xsi,zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm)
	eta = m1*m2/pow(m1+m2,2.)
	fcut_mu0 = 0.3236+0.04894*xsi+0.01346*xsi*xsi
	fcut_y = -0.1331*eta -0.2714*eta*eta +4.922*eta*eta*eta - 0.08172*eta*xsi +0.1451*eta*xsi*xsi +0.1279*eta*eta*xsi
	return (fcut_mu0+fcut_y)/(math.pi*mtot)

def Search_Omg(Omega, freq_ref):
    interp = InterpolatedUnivariateSpline(GS.Freq, Omega)
    out = np.zeros(len(freq_ref))
    for i in range(len(freq_ref)) :
        out[i] = interp(freq_ref[i])
    return out
import numpy as np
import math

def fcut_f(m1, m2, xsi, zm):
	m1, m2, xsi, zm = np.asarray(m1), np.asarray(m2), np.asarray(xsi), np.asarray(zm)

	mtot = (m1 + m2) * 4.9685e-6 * (1 + zm)
	eta = m1 * m2 / np.power(m1 + m2, 2.)

	fcut_mu0 = 0.3236 + 0.04894 * xsi + 0.01346 * xsi ** 2
	fcut_y = (-0.1331 * eta - 0.2714 * eta ** 2 + 4.922 * eta ** 3
			  - 0.08172 * eta * xsi + 0.1451 * eta * xsi ** 2 + 0.1279 * eta ** 2 * xsi)

	result = (fcut_mu0 + fcut_y) / (math.pi * mtot)

	# S'assurer que le retour est bien un float si les entrées sont scalaires
	if np.isscalar(m1) and np.isscalar(m2) and np.isscalar(xsi) and np.isscalar(zm):
		return float(result)
	return result.astype(float)

def zmaximal(m1,m2,xsi,fmin) :
	mtot = (m1+m2)*4.9685e-6
	eta = m1*m2/pow(m1+m2,2.)
	fmerg_mu0 = 1.-4.455*pow(1-xsi,0.217)+3.521*pow(1.-xsi,0.26)
	fmerg_y = 0.6437*eta -0.05822*eta*eta -7.092*eta*eta*eta +0.827*eta*xsi -0.2706*eta*xsi*xsi -3.935*eta*eta*xsi
	return (fmerg_mu0+fmerg_y)/(math.pi*mtot*fmin)-1

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
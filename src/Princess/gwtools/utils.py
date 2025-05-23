import numpy as np
import pandas as pd
import math
from scipy.special import jv
from Princess.Run.Constants import *

def fcut_f(m1, m2, xsi, zm):
	m1, m2, xsi, zm = np.asarray(m1), np.asarray(m2), np.asarray(xsi), np.asarray(zm)

	mtot = (m1 + m2) * 4.9685e-6 * (1 + zm) # conversion Msun to grams
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
	mtot = (m1+m2)*4.9685e-6 # conversion Msun to grams
	eta = m1*m2/pow(m1+m2,2.)
	fmerg_mu0 = 1.-4.455*pow(1-xsi,0.217)+3.521*pow(1.-xsi,0.26)
	fmerg_y = 0.6437*eta -0.05822*eta*eta -7.092*eta*eta*eta +0.827*eta*xsi -0.2706*eta*xsi*xsi -3.935*eta*eta*xsi
	return (fmerg_mu0+fmerg_y)/(math.pi*mtot*fmin)-1

def fmerg_f(m1,m2,xsi,zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm) # conversion Msun to grams
	eta = m1*m2/pow(m1+m2,2.)
	fmerg_mu0 = 1.-4.455*pow(1-xsi,0.217)+3.521*pow(1.-xsi,0.26)
	fmerg_y = 0.6437*eta -0.05822*eta*eta -7.092*eta*eta*eta +0.827*eta*xsi -0.2706*eta*xsi*xsi -3.935*eta*eta*xsi
	return (fmerg_mu0+fmerg_y)/(math.pi*mtot)

def fring_f(m1,m2,xsi,zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm) # conversion Msun to grams
	eta = m1*m2/pow(m1+m2,2.)
	fring_mu0 = (1.-0.63*pow(1.-xsi,0.3))/2.
	fring_y = 0.1469*eta -0.0249*eta*eta +2.325*eta*eta*eta - 0.1228*eta*xsi -0.02609*eta*xsi*xsi +0.1701*eta*eta*xsi
	return (fring_mu0+fring_y)/(math.pi*mtot)

def sigma_f(m1,m2,xsi,zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm) # conversion Msun to grams
	eta = m1*m2/pow(m1+m2,2.)
	sigma_mu0 =(1.-0.63*pow(1.-xsi,0.3))*pow(1.-xsi,0.45)/4.
	sigma_y = -0.4098*eta +1.829*eta*eta -2.87*eta*eta*eta - 0.03523*eta*xsi +0.1008*eta*xsi*xsi -0.02017*eta*eta*xsi
	return (sigma_mu0+sigma_y)/(math.pi*mtot)


# Under developpement
# def objective(trial):
#     """Objective function for Optuna optimization."""
#     params = {
#         "learning_rate": trial.suggest_float("learning_rate", 0.075, 0.09),
#         "max_iter": trial.suggest_int("max_iter", 290, 320),  # Number of trees
#         "max_depth": trial.suggest_int("max_depth", 22, 37),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 91, 94),
#         "l2_regularization": trial.suggest_float("l2_regularization", 2.02, 2.175),
#         "max_leaf_nodes" : trial.suggest_int("max_leaf_nodes", 160, 167),
#         "max_features" : trial.suggest_float("max_features", 0.6, 0.9)
#     }
#
#     # Train the model
#     model = HistGradientBoostingRegressor(**params)
#     model.fit(train_X1, train_L1)
#
#     # Predict on validation set
#     y_pred = model.predict(test_X1)
#
#     # Evaluate performance (Minimize MAE)
#     mae = mean_absolute_error(test_L1, y_pred)
#
#     # Store trial results
#     results_list.append({**params, "mae": mae})
#
#     return mae


def Mc(m1, m2):
    """This function does the mapping (m1,m2) --> (mc,q)

    Parameters
    ----------
    m1 : float or numpy array
        Mass of primary of the source(s) in Msun
    m2 : float or numpy array
        Mass of seconday of the source(s) in Msun

    Returns
    -------
    mc : float or numpy array
        Chirp mass of the sources(s) in Msun
    q : float or numpy array
        Mass ratio of the source(s)
    """

    mc = np.power((m1*m2), 0.6) / (np.power(m1 + m2, 0.2))

    return mc

def compute_ecc_impact():
	'''Compute and save Gne and Psie the factors of eccentricity impactin teh  GW energy density.'''
	e_arr = np.logspace(-6, 0, 5000)

	psi_e = pd.DataFrame({'e': e_arr})
	g_ne = pd.DataFrame({'e': e_arr})
	factor = pd.DataFrame({'e': e_arr})

	for m in range(20):
		n= m+2
		epsilon = 0 #1.e-15  # Petit terme pour éviter division par zéro

		# psi_e
		denom_psi = np.power(1 - e_arr * e_arr, 7. / 2.) + epsilon
		psi_e[f'n={str(n)}'] = (1 + 73. * e_arr ** 2 / 74. + 37. * e_arr ** 4 / 96.) / denom_psi

		# g_ne
		A = jv(n - 2, n * e_arr) - 2 * e_arr * jv(n - 1, n * e_arr) + 2 / n * jv(n, n * e_arr) \
			+ 2 * e_arr * jv(n + 1, n * e_arr) - jv(n + 2, n * e_arr)

		B = jv(n - 2, n * e_arr) - 2 * jv(n, n * e_arr) + jv(n + 2, n * e_arr)

		C = jv(n, n * e_arr)

		g_ne[f'n={str(n)}'] = (
				np.power(4. / (n * n + epsilon), 1. / 3.) *
				np.power(n, 4.) / 32 *
				(np.power(A, 2.) +
				 (1 - e_arr * e_arr) * np.power(B, 2.) +
				 4 / (3 * (n * n + epsilon)) * C * C)
		)

		# factor
		denom_factor = psi_e[f'n={str(n)}'] + epsilon
		factor[f'n={str(n)}'] = np.power(4. / (n * n + epsilon), 1. / 3.) * g_ne[f'n={str(n)}'] / denom_factor

	psi_e.to_csv('./AuxiliaryFiles/eccentricity_impact/psi_e_n.dat', sep = '\t', index = None)
	g_ne.to_csv('./AuxiliaryFiles/eccentricity_impact/g_ne.dat', sep = '\t', index = None)
	factor.to_csv('./AuxiliaryFiles/eccentricity_impact/factor_ecc.dat', sep = '\t', index = None)

def eccentricity_evolution(event, e0, f0):
	''' Computation of eccentricity evolution with the orbital frequency.
	:param e0: initial eccentricity
	:param f0: initial orbital frequency
	:return: arrayys with orbital frequency and corresponding eccentricity
	'''
	flso = fcut_f(m1=event['m1'], m2=event['m2'], xsi=0, zm=event['z'])
	f0 = orbital_frequency(m1 = event['m1'], m2 = event['m2'], a = event['a0'])
	forb = np.logspace(max(np.log(f0), -6), min(0,np.log(flso)), 200)
	freq = forb/f0
	e = e0 * np.power(freq, -19./18.) * (1 +	3323/1824 * e0**2 (1 - np.power(freq, -19/9)) +
		15994231/6653952 * e0**4 * ( 1 - 66253974/15994231 * np.power(freq, -19./9.) + 50259743/15994231 * np.power(freq, -38./9.)) +
		105734339801/36410425344 * e0 **6 ( 1 -	1138825333323/105734339801 * np.power(freq,-19./9.) + 2505196889835/105734339801 * np.power(freq, -38./9.) -
		1472105896313/105734339801 * np.power(freq, -19./3.)))
	return forb, e


def interpolate_with_zeros_outside(x, y, x_target):
    """
    Interpolates y(x) at the positions x_target.
    Returns 0 for any values in x_target that fall outside the range of x.

    Parameters:
    - x : array-like, known x values (must be sorted in ascending order)
    - y : array-like, corresponding y values
    - x_target : array-like, target x positions to interpolate at

    Returns:
    - y_target : np.ndarray, interpolated y values at x_target
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_target = np.asarray(x_target)

    # Perform linear interpolation, returning 0 outside the range of x
    y_interp = np.interp(x_target, x, y, left=0, right=0)

    return y_interp

def orbital_frequency(m1, m2, a):
	"""
	Computes the orbital frequency of the binary system in the source frame from the third Keplerian law.
	:param m1: Primary objetc mass in Msun
	:param m2: Secondary objetc mass in Msun
	:param a0: Semi major axis in Rsun
	:return: Orbital frequency in Hz
	"""
	forb = 1/(2*math.pi) * np.sqrt(G * M_SUN * (m1+m2) / (a * R_SUN)**3)
	return forb


import os
import math
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import pandas as pd
from scipy.optimize import root
from matplotlib.patches import Patch
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import Planck15

def toCartesian(r,theta,phi):
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]

def toSpherical(x,y,z):
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)
    phi     =  np.arctan(y/x)
    return [r,theta,phi]

def DetFramePosition(r,theta, phi, Mat, t):
	pos_earthFrame_cart= toCartesian(r,theta,phi)
	pos_detFrame_cart = np.matmul(Mat, pos_earthFrame_cart) - t
	pos_det_sph = toSpherical(pos_detFrame_cart[0], pos_detFrame_cart[1], pos_detFrame_cart[2])
	return [pos_det_sph[1],pos_det_sph[2]]

def Rotation(theta, Mat):
	return np.dot(Mat,np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta),0], [0,0,1]]))

def Factor(inc, theta, phi, epsilon):
	return(np.power(np.sin(epsilon)*(1+inc*inc)*(1+np.cos(theta)*np.cos(theta))*np.cos(2*phi)/2.,2)+np.power(np.sin(epsilon)*inc*np.cos(theta)*np.sin(2*phi),2.))


#Detectors placement from https://lscsoft.docs.ligo.org/lalsuite/lal/group___detector_constants.html
#ET1 placed at the current Virgo interferometer, with aligned Xarms.

#LIGO-Hanford
#define LAL_LHO_4K_VERTEX_LOCATION_X_SI         -2.16141492636e+06      /**< LHO_4k x-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_4K_VERTEX_LOCATION_Y_SI         -3.83469517889e+06      /**< LHO_4k y-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_4K_VERTEX_LOCATION_Z_SI         4.60035022664e+06       /**< LHO_4k z-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_4K_ARM_X_DIRECTION_X            -0.22389266154  /**< LHO_4k x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_X_DIRECTION_Y            0.79983062746   /**< LHO_4k y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_X_DIRECTION_Z            0.55690487831   /**< LHO_4k z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_Y_DIRECTION_X            -0.91397818574  /**< LHO_4k x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_Y_DIRECTION_Y            0.02609403989   /**< LHO_4k y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_Y_DIRECTION_Z            -0.40492342125  /**< LHO_4k z-component of unit vector pointing along y arm in Earth-centered frame */

#LIGO-Livingstone
#define LAL_LLO_4K_VERTEX_LOCATION_X_SI         -7.42760447238e+04      /**< LLO_4k x-component of vertex location in Earth-centered frame (m) */
#define LAL_LLO_4K_VERTEX_LOCATION_Y_SI         -5.49628371971e+06      /**< LLO_4k y-component of vertex location in Earth-centered frame (m) */
#define LAL_LLO_4K_VERTEX_LOCATION_Z_SI         3.22425701744e+06       /**< LLO_4k z-component of vertex location in Earth-centered frame (m) */
#define LAL_LLO_4K_ARM_X_DIRECTION_X            -0.95457412153  /**< LLO_4k x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_X_DIRECTION_Y            -0.14158077340  /**< LLO_4k y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_X_DIRECTION_Z            -0.26218911324  /**< LLO_4k z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_Y_DIRECTION_X            0.29774156894   /**< LLO_4k x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_Y_DIRECTION_Y            -0.48791033647  /**< LLO_4k y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_Y_DIRECTION_Z            -0.82054461286  /**< LLO_4k z-component of unit vector pointing along y arm in Earth-centered frame */

#Virgo
#define LAL_VIRGO_VERTEX_LOCATION_X_SI          4.54637409900e+06       /**< VIRGO x-component of vertex location in Earth-centered frame (m) */
#define LAL_VIRGO_VERTEX_LOCATION_Y_SI          8.42989697626e+05       /**< VIRGO y-component of vertex location in Earth-centered frame (m) */
#define LAL_VIRGO_VERTEX_LOCATION_Z_SI          4.37857696241e+06       /**< VIRGO z-component of vertex location in Earth-centered frame (m) */
#define LAL_VIRGO_ARM_X_DIRECTION_X             -0.70045821479  /**< VIRGO x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_VIRGO_ARM_X_DIRECTION_Y             0.20848948619   /**< VIRGO y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_VIRGO_ARM_X_DIRECTION_Z             0.68256166277   /**< VIRGO z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_VIRGO_ARM_Y_DIRECTION_X             -0.05379255368  /**< VIRGO x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_VIRGO_ARM_Y_DIRECTION_Y             -0.96908180549  /**< VIRGO y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_VIRGO_ARM_Y_DIRECTION_Z             0.24080451708   /**< VIRGO z-component of unit vector pointing along y arm in Earth-centered frame */

#KAGRA
#define LAL_KAGRA_VERTEX_LOCATION_X_SI          -3777336.024       /**< KAGRA x-component of vertex location in Earth-centered frame (m) */
#define LAL_KAGRA_VERTEX_LOCATION_Y_SI          3484898.411       /**< KAGRA y-component of vertex location in Earth-centered frame (m) */
#define LAL_KAGRA_VERTEX_LOCATION_Z_SI          3765313.697       /**< KAGRA z-component of vertex location in Earth-centered frame (m) */
#define LAL_KAGRA_ARM_X_DIRECTION_X             -0.3759040  /**< KAGRA x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_KAGRA_ARM_X_DIRECTION_Y              -0.8361583   /**< KAGRA y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_KAGRA_ARM_X_DIRECTION_Z             0.3994189   /**< KAGRA z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_KAGRA_ARM_Y_DIRECTION_X              0.7164378  /**< KAGRA x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_KAGRA_ARM_Y_DIRECTION_Y              0.01114076  /**< KAGRA y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_KAGRA_ARM_Y_DIRECTION_Z             0.6975620   /**< KAGRA z-component of unit vector pointing along y arm in Earth-centered frame */


K = dict({'a1':[-0.3759040,-0.8361583,0.3994189],
		  'a2':[0.7164378,0.01114076,0.6975620],
		  'z':[-0.58772209,  0.5483751, 0.5948676],
		  'earth_frame_pos': [4.54637409900e+06, 8.42989697626e+05,  4.37857696241e+06] })
M_Kagra = np.array([K['a1'],K['a2'],np.cross(K['a1'],K['a2'])])

H = dict({'a1':[-0.22389266154,0.79983062746,0.55690487831], #a1 corrspond to the x-arm
		  'a2':[-0.91397818574,0.02609403989,-0.40492342125], #a2 is the second y-arm
		  'z': [-0.33840205, -0.59965829,  0.72518548], #z is the perpendicular to a1 and a2
		  'earth_frame_pos': [-2.16141492636e+06, -3.83469517889e+06, 4.60035022664e+06] })#earth_frame_pos is the position od the vertex (beamsplitter) of the detector in the earth frame.
print(np.cross(H['a1'],H['a2']))
M_Hanford = np.array([H['a1'],H['a2'],H['z']])

L = dict({'a1':[-0.95457412153,-0.14158077340,-0.26218911324],
		  'a2':[0.29774156894,-0.48791033647,-0.82054461286],
		  'z':[-0.01175144, -0.86133525,  0.50790106],
		  'earth_frame_pos': [-7.42760447238e+04, -5.49628371971e+06,  3.22425701744e+06] })
M_Livingtone = np.array([L['a1'],L['a2'],L['z']])


V = dict({'a1':[-0.70045821479,0.20848948619,0.68256166277],
		  'a2':[-0.05379255368,-0.96908180549,0.24080451708],
		  'z':[0.7116633,  0.13195677, 0.69001649],
		  'earth_frame_pos': [4.54637409900e+06, 8.42989697626e+05,  4.37857696241e+06] })
M_Virgo = np.array([V['a1'],V['a2'],V['z']])

ET1 = dict({'a1':[-0.70045821479,0.20848948619,0.68256166277],
		  'a2':[-0.39681482542, -0.73500471881, 0.54982366052],
		  'z':np.cross([-0.70045821479,0.20848948619,0.68256166277],[-0.39681482542, -0.73500471881, 0.54982366052]),
		  'earth_frame_pos': [4.54637409900e+06, 8.42989697626e+05,  4.37857696241e+06] })
M_ET1 = np.array([ET1['a1'],ET1['a2'],ET1['z']])

ET2 = dict({'a1':[0.30364338937, -0.94349420500,-0.13273800225],
		  'a2':[0.70045821479, -0.20848948619, -0.68256166277],
		  'z': np.cross([0.30364338937, -0.94349420500,-0.13273800225],[0.70045821479, -0.20848948619, -0.68256166277]),
		  'earth_frame_pos': [4.53936951685e+06, 8.45074592488e+05,  4.38540257904e+06] })
M_ET2 = np.array([ET2['a1'],ET2['a2'],ET2['z']])

ET3 = dict({'a1':[0.39681482542,0.73500471881, -0.54982366052],
		  'a2':[-0.30364338937,0.94349420500, 0.13273800225],
		  'z':np.cross([0.39681482542,0.73500471881, -0.54982366052],[-0.30364338937,0.94349420500, 0.13273800225]),
		  'earth_frame_pos': [4.54240595075e+06,  8.35639650438e+05,  4.38407519902e+06] })
M_ET3 = np.array([ET3['a1'],ET3['a2'],ET3['z']])

I = dict({'a1':[0.38496278183, -0.39387275094,0.83466634811],
		  'a2':[0.89838844906, -0.04722636126,-0.43665531647],
		  'z': np.cross([0.38496278183, -0.39387275094,0.83466634811],[0.89838844906, -0.04722636126,-0.43665531647]),
		  'earth_frame_pos': [1.34897115479e+06, 5.85742826577e+06,  2.12756925209e+06] })
M_Ligo_India = np.array([I['a1'],I['a2'],I['z']])


#M_ET1 =  M_Virgo
#M_ET2 = Rotation(math.pi/3., M_Virgo)
#M_ET3 = Rotation(2*math.pi/3., M_Virgo)


r=np.sqrt(3./2.)
r= 1./np.sqrt(2.)
theta = math.pi/2.
phi = math.pi/4.



Mat = np.array([[1.,1.,0],[-1.,1.,0],[0,0,1]])
t =np.array([0,0,0])

r = np.sqrt(2)
theta = math.pi/2.
phi = math.pi/4.
DetFramePosition(r,theta,phi,Mat,t)


factorH = np.empty(100000)
factorL = np.empty(100000)
factorV = np.empty(100000)
factorK = np.empty(100000)
factorET1 = np.empty(100000)
factorET2 = np.empty(100000)
factorET3 = np.empty(100000)
factorI = np.empty(100000)
INC = np.empty(100000)
THETA = np.empty(100000)
PHI = np.empty(100000)

for i in range(100000) : 
	r=100.*3.1e22
	inc = np.random.uniform(-1.,1.)
	theta = np.random.uniform(0.,2*math.pi)
	phi = np.random.uniform(0,math.pi)
	INC[i]= inc
	PHI[i]= phi
	THETA[i]= theta

	H_frame = DetFramePosition(r,theta,phi,M_Hanford,H['earth_frame_pos'])
	factorH[i] = np.sqrt(Factor(inc,H_frame[0],H_frame[1], math.pi/2.))/2.

	L_frame = DetFramePosition(r,theta,phi,M_Livingtone,L['earth_frame_pos'])
	factorL[i] = np.sqrt(Factor(inc,L_frame[0],L_frame[1], math.pi/2.))/2.

	V_frame = DetFramePosition(r,theta,phi,M_Virgo,V['earth_frame_pos'])
	factorV[i] = np.sqrt(Factor(inc,V_frame[0],V_frame[1], math.pi/2.))/2.

	K_frame = DetFramePosition(r,theta,phi,M_Kagra,K['earth_frame_pos'])
	factorK[i] = np.sqrt(Factor(inc,K_frame[0],K_frame[1], math.pi/2.))/2.

	I_frame = DetFramePosition(r, theta, phi, M_Ligo_India, I['earth_frame_pos'])
	factorI[i] = np.sqrt(Factor(inc, I_frame[0], I_frame[1], math.pi / 2.)) / 2.

	ET1_frame = DetFramePosition(r,theta,phi,M_ET1,ET1['earth_frame_pos'])
	ET2_frame = DetFramePosition(r,theta,phi,M_ET2,ET2['earth_frame_pos'])
	ET3_frame = DetFramePosition(r,theta,phi,M_ET3,ET3['earth_frame_pos'])

	factorET1[i] = np.sqrt(Factor(inc,ET1_frame[0],ET1_frame[1], math.pi/3.))/2.
	factorET2[i] = np.sqrt(Factor(inc,ET2_frame[0],ET2_frame[1], math.pi/3.))/2.
	factorET3[i] = np.sqrt(Factor(inc,ET3_frame[0],ET3_frame[1], math.pi/3.))/2.

output = pd.DataFrame({'inc': INC, 'phi':PHI, 'theta': THETA,'H': factorH, 'L':factorL,'V':factorV, 'K':factorK, 'E1':factorET1, 'E2':factorET1, 'E3':factorET1})
output.to_csv('factor_table.dat', sep = '\t', index = None)
print('factor table saved!')








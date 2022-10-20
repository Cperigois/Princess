import math

""" All astrophysical constant needed are defined her :)"""

G = 6.674e-8  # cm3 g-1 s-2
M_sun = 1.99e33  # g
H0 = 2.183e-18  # s-1
Mpc = 3.086e24  # cm
yr = 365 * 24 * 3600  # s
deltaT = 10  # Myr
c = 2.99e10  # cm s-1
R_sun = 6.955e10  # cm
pi = math.pi
R_earth = 6371.e5 #cm

K1 = 5. * pow(math.pi, 2. / 3.) * pow(G, 5. / 3.) / (yr * 18. * pow(c, 3.) * H0 * H0)  # Mdc formule
# K1 = 2.*pow(pi,2./3.)*pow(G,5./3.)/(yr*9.*pow(c,3.)*H0*H0)*(5./4.) # including inclinason factor

rhoc = 3. * c * c * H0 * H0 / (8 * math.pi * G)
Cst = 256. * pow(math.pi, 8. / 3.) * pow(G, 5. / 3.) / (5 * pow(c, 5.))


C = pi*c*c/(rhoc*2.*G*yr)

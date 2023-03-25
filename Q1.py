import numpy as np
import matplotlib.pyplot as plt

# Constants

GM=3.986E5 # km^3/s^2 for Earth
R = 6370 # km Earth Radius
J2 = 0.00108 # For Earth
T=1/3.0*3600*24
n = 360.0/T
deg2rad = np.pi/180.0
rad2deg = 1/deg2rad

def Omegadot(a,i,e):
    '''
    Function calculating the Omega\bar\dot: Nodal precession
    :param n: Satellite's mean motion in degree/s
    :param a: Semimajor axis of orbit in km
    :param i: Inclination in degree
    :param e: Eccentricity of ellipse orbit
    :return: Omega\bar\dot in degree/s
    '''

    i *= deg2rad

    return -3/2.0*n*J2*(R/a)**2*np.cos(i)/(1-e**2)**2

# According to satisfy equation (1), we have:
the_i = np.arccos((1/5.0)**0.5)*rad2deg

print("Inclination i=%.2f°"%the_i)

# According the Kepler 3rd law, we have:

the_a = (GM*T**2/4/np.pi**2)**(1/3)

print("Semimajor axis a=%.2f km"%the_a)

# Minimum e given 600km minimum perigee altitude
minPerigee = 600
min_e = 1-(minPerigee+R)*2/the_a

print("Minimum eccentricity e=%.2f"%min_e)

# Plot the change in Omega\bar\dot to find the minimum value:

es = np.linspace(min_e,0.85,1000)

fig = plt.figure(figsize=(7,3))
ax = fig.add_subplot(1,1,1)

ax.plot(es,Omegadot(the_a,the_i,es))
ax.grid()

ax.set_xlabel("Eccentricity  e")
ax.set_ylabel("$\\dot{\\bar{\Omega}}$ (°/s)")

plt.show()

print("Lowest change in Omegadotbar = %.2e °/s" % Omegadot(the_a,the_i,min_e))



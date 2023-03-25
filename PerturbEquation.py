import numpy as np
import matplotlib.pyplot as plt

J2 = 0.00108
mu = 3.986e5  # km3/s2
R = 6370  # km
deg2rad = np.pi/180.0
rad2deg = 1/deg2rad

Nt = 100000
tall = 2 * 3600 * 24 # Days to second

# J2 perturbations of Earth
def pert_J2_rsw(r, i, omega, theta):
    '''
    Based on (10.88) in Curtis, 2020
    :param r: radius location of object in km
    :param i: inclination in degree
    :param omega: Argument of perigee in degree
    :param theta: true anomaly in degree
    :return: [pr,ps,pw] of J2 perturbation
    '''

    # Turning degrees to rads
    i = i *deg2rad
    omega = omega*deg2rad
    theta = theta*deg2rad

    # Based on (10.88) in Curtis, 2020
    pr = -3/2.0*J2*mu*R**2/r**4*(1-3*np.sin(i)**2*np.sin(omega+theta)**2)
    ps = -3/2.0*J2*mu*R**2/r**4*np.sin(i)**2*np.sin(2*omega+2*theta)
    pw = -3/2.0*J2*mu*R**2/r**4*np.sin(2*i)*np.sin(omega+theta)

    return np.array([pr,ps,pw])

# Gauss planetary equation to get perturbations
def GaussPlanetEqn(h,e,theta,Omega,i,omega,p_vec):
    '''
    The Gauss form of planetary equation, following Curtis (2020), Eq. 10.84
    :param h: Angular momentum
    :param e: Eccentricity
    :param theta: true anomaly in degree
    :param Omega: Node angle
    :param i: inclination
    :param omega: argument of periapsis
    :param p_vec: perturbation in rsw coordinate
    :return:
    '''

    pr,ps,pw = p_vec

    # Turning degrees to rads
    i = i *deg2rad
    omega = omega*deg2rad
    theta = theta*deg2rad
    Omega = Omega*deg2rad

    # a = h**2/mu, r = a(1+ecos(theta))
    r = h**2/(mu*(1+e*np.cos(theta)))
    # a = h**2/mu

    # The Gauss form of planetary equation, following Curtis (2020), Eq. 10.84
    dhdt = r*ps
    dedt = h/mu*np.sin(theta)*pr + 1/mu/h*((h**2+mu*r)*np.cos(theta)+mu*e*r)*ps
    dthedt = h/r**2 + 1/e/h*(h**2/mu*np.cos(theta)*pr - (r+h**2/mu)*np.sin(theta)*ps)
    dOmedt = r/h/np.sin(i)*np.sin(omega+theta) * pw
    didt = r/h*np.cos(omega+theta) * pw
    domedt = -1/e/h*(h**2/mu*np.cos(theta)*pr-(r+h**2/mu)*np.sin(theta)*ps) - r*np.sin(omega+theta)/h/np.tan(i)*pw

    return np.array([dhdt,dedt,dthedt*rad2deg,dOmedt*rad2deg,didt*rad2deg,domedt*rad2deg])

# Convert from mean anomaly to true anomaly
def M2theta(M,e):

    # Check if M=0/360:
    M = np.mod(M,360)
    if M==0:
        return 0.0

    M = M*deg2rad

    # 1.  Solve the equation from M to E: E-esin(E)=M
    E = M
    # Newton's method
    Nmax = 1000
    for i in range(Nmax):

        E1 = E-(M-E+e*np.sin(E))/(-1+e*np.cos(E))
        if np.abs((E1-E)/E1)<1e-8:
            E = E1
            break
        else:
            E = E1

    if i == Nmax-1:
        raise BrokenPipeError("Not converge!@")

    # Convert E to theta
    theta = 2*np.arctan(np.tan(E/2.0)*((1+e)/(1-e))**0.5)

    return theta*rad2deg

# Convert true anomaly to mean anomaly
def theta2M(theta,e):

    theta = theta * deg2rad
    M = 2*np.arctan((((1-e)/(1+e))**0.5*np.tan(theta/2)))-e*(1-e**2)**0.5*np.sin(theta)/(1+e*np.cos(theta)) # Curtis 2020, Eq.3.6
    M = M*rad2deg

    return M


# Initial conditions
a_0 = 26600 # km
i_0 = 63.43 # 1.10654 / np.pi*180 # Degree
e_0 = 0.74
omega_0 = 5 # Degree
Omega_0 = 90 # Degree
M_0 = 10 # Degree

# Below are two sets of testing parameters used to test the program.
# a_0 = 8309 # km
# i_0 = 28 # Degree
# e_0 = 0.196
# omega_0 = 30 # Degree
# Omega_0 = 45 # Degree
# M_0 = 40 # Degree

# a_0 = 8059 # km
# i_0 = 28 # Degree
# e_0 = 0.17136
# omega_0 = 30 # Degree
# Omega_0 = 45 # Degree
# M_0 = 40 # Degree

# a_0 = 7000 # km
# i_0 = 0 # Degree
# e_0 = 0.05
# omega_0 = 0 # Degree
# Omega_0 = 0 # Degree
# M_0 = 0 # Degree


# Set time array
ts = np.linspace(0,tall,Nt)

data = [np.array([a_0,e_0,i_0,omega_0,Omega_0,M_0])]

# Advect the perturbation in orbital elements
for t in range(1,len(ts)):

    dt = ts[t] - ts[t-1]

    a,e,i,omega,Omega,M = data[-1]
    h = (a * mu * (1-e**2)) ** 0.5 # Get angular momentum
    theta = M2theta(M,e) # Get new theta
    r = h ** 2 / (mu * (1 + e * np.cos(theta*deg2rad))) # Get distance

    # Calculate perturbation
    p_vec = pert_J2_rsw(r, i, omega, theta)

    # Input into Planetary equation
    dhdt,dedt,dthedt,dOmedt,didt,domedt = GaussPlanetEqn(h, e, theta, Omega, i, omega, p_vec)

    # Simple way to advect changes into the orbital elements
    h1 = h+dhdt*dt
    e1 = e+dedt*dt
    theta1 = theta+dthedt*dt
    Omega1 = Omega+dOmedt*dt
    i1 = i+didt*dt
    omega1 = omega+domedt*dt

    # h1 = h+dhdt*0
    # e1 = e+dedt*0
    # theta1 = theta+dthedt*0
    # Omega1 = Omega+dOmedt*0
    # i1 = i+didt*0
    # omega1 = omega+domedt*0

    a1 = h1**2/mu/(1-e1**2) # Get new semimajor axis from new angular momentum.

    M1 = theta2M(theta1,e1) # Return the true anomaly to mean anomaly.

    # Calculate new mean anomaly
    # M1 = M_pert + mu**2/h**3*(1-e**2)**(3/2.0)*dt*rad2deg

    # Reset M1 when necessary to limit it in 360 degree.
    if M1>360:
        M1-=360

    # Append result to the main data array
    data.append([a1,e1,i1,omega1,Omega1,M1])

data = np.array(data)

fig,axs = plt.subplots(5,1,figsize=(13,5*5))

ax = axs[0]
ax.plot(ts/3600/24,data[:,0])
ax.set_title("Semimajor Axis a")
ax.grid()
ax.set_xlabel("Time [days]")
ax.set_ylabel("a [km]")

ax = axs[1]
ax.plot(ts/3600/24,data[:,1])
ax.set_title("Eccentricity e")
ax.grid()
ax.set_xlabel("Time [days]")
ax.set_ylabel("e")

ax = axs[2]
ax.plot(ts/3600/24,data[:,2])
ax.set_title("Inclination i")
ax.grid()
ax.set_xlabel("Time [days]")
ax.set_ylabel("i [deg]")

ax = axs[3]
ax.plot(ts/3600/24,data[:,3])
ax.set_title("Argument of periapsis $\omega$")
ax.grid()
ax.set_xlabel("Time [days]")
ax.set_ylabel("$\omega$ [deg]")

ax = axs[4]
ax.plot(ts/3600/24,data[:,4])
ax.set_title("Node angle $\Omega$")
ax.grid()
ax.set_xlabel("Time [days]")
ax.set_ylabel("$\Omega$ [deg]")
plt.show()
# print(data[:,5])












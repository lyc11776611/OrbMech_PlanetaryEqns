import numpy as np
import matplotlib.pyplot as plt

J2 = 0.00108
mu = 3.986e5  # km3/s2
R = 6370  # km
deg2rad = np.pi/180.0
rad2deg = 1/deg2rad

Nt = 100000
tall = 10 * 3600 * 24 # Days to second

plotelems = 1 # 0 for aeioom, 1 for ahkpq \lambda

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
def GaussPlanetEqn(a,h,k,p,q,L,p_vec):
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
    L = L *deg2rad

    # a = h**2/mu, r = a(1+ecos(theta))
    # r = h**2/(mu*(1+e*np.cos(theta)))
    # a = h**2/mu

    s = (1-k**2-h**2)**0.5
    W = 1+k*np.cos(L)+h*np.sin(L)
    A = k + np.cos(L) * (1+W)
    B = h + np.sin(L) * (1+W)
    slr = a*(1-h**2-k**2)
    X = 1 + p**2 + q**2


    # The Gauss form of planetary equation, following Curtis (2020), Eq. 10.84
    dadt = 2*(a/mu)**0.5*a/s*((k*np.sin(L)-h*np.cos(L))*pr+W*ps)
    dhdt = (slr/mu)**0.5/W*(-W*np.cos(L)*pr + B*ps + k*(q*np.sin(L)-p*np.cos(L))*pw)
    dkdt = (slr/mu)**0.5/W*(W*np.sin(L)*pr + A*ps - h*(q*np.sin(L)-p*np.cos(L))*pw)
    dqdt = 0.5*(slr/mu)**0.5*X/W*np.cos(L)*pw
    dpdt = 0.5*(slr/mu)**0.5*X/W*np.sin(L)*pw
    dLdt = (mu/slr**3)**0.5*W**2 + (slr/mu)**0.5/W*(q*np.sin(L)-p*np.cos(L))*pw

    return np.array([dadt,dhdt,dkdt,dqdt,dpdt,dLdt*rad2deg])

# Convert from mean anomaly to true anomaly
def M2theta(M,h,k):

    e = (h**2+k**2)**0.5
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
def theta2M(theta,h,k):
    e = (h**2+k**2)**0.5
    theta = theta * deg2rad
    M = 2*np.arctan((((1-e)/(1+e))**0.5*np.tan(theta/2)))-e*(1-e**2)**0.5*np.sin(theta)/(1+e*np.cos(theta)) # Curtis 2020, Eq.3.6
    M = M*rad2deg

    return M

def aeioom2ahkpql(obj):
    a, e, i, omega, Omega, M = obj

    omega = omega * deg2rad
    Omega = Omega * deg2rad
    i = i * deg2rad

    h = e*np.sin(omega+Omega)
    k = e*np.cos(omega+Omega)

    p = np.tan(i/2)*np.sin(Omega)
    q = np.tan(i/2)*np.cos(Omega)

    theta = M2theta(M,h,k)

    L = theta+omega*rad2deg+Omega*rad2deg

    return [a,h,k,p,q,L]

def ahkpql2aeioom(obj):
    a,h,k,p,q,L = obj

    e = (h**2+k**2)**0.5
    i = np.arctan((p**2+q**2)**0.5)*2

    # Avoid nans when inclination is 0 for omegabar = Omega+omega
    if k==0:
        omegabar = 0
    else:
        omegabar = np.arctan(h/k)
        if k<0:
            omegabar += np.pi


    if q==0:
        Omega = 0
    else:
        Omega = np.arctan(p/q)
        if q/np.tan(i/2.0)<0:
            Omega += np.pi

    omega = omegabar - Omega

    omegabar = omegabar * rad2deg
    i = i * rad2deg
    Omega = np.mod(Omega * rad2deg,360)
    omega = np.mod(omega * rad2deg,360)

    M = theta2M(L-omegabar,h,k)

    return [a, e, i, omega, Omega, M]


# Initial conditions
# a_0 = 26600 # km
# i_0 = 63.43 # 1.10654 / np.pi*180 # Degree
# e_0 = 0.74
# omega_0 = 5 # Degree
# Omega_0 = 90 # Degree
# M_0 = 10 # Degree

# Below are two sets of testing parameters used to test the program.
# a_0 = 8309 # km
# i_0 = 28 # Degree
# e_0 = 0.196
# omega_0 = 30 # Degree
# Omega_0 = 45 # Degree
# M_0 = 40 # Degree

a_0 = 7000 # km
i_0 = 0 # Degree
e_0 = 0.05
omega_0 = 0 # Degree
Omega_0 = 0 # Degree
M_0 = 0 # Degree

# Set time array
ts = np.linspace(0,tall,Nt)

data = [np.array([a_0,e_0,i_0,omega_0,Omega_0,M_0])]
data1 = [aeioom2ahkpql([a_0,e_0,i_0,omega_0,Omega_0,M_0])]
lambds = [M_0+omega_0+Omega_0]

# Advect the perturbation in orbital elements
for t in range(1,len(ts)):

    dt = ts[t] - ts[t-1]

    a,h,k,p,q,L = data1[-1]
    a, e, i, omega, Omega, M = data[-1]

    theta = np.mod(L-omega-Omega,360)
    angmot = (a * mu * (1 - e ** 2)) ** 0.5  # Get angular momentum

    r = angmot ** 2 / (mu * (1 + e * np.cos(theta*deg2rad))) # Get distance

    # Calculate perturbation
    p_vec = pert_J2_rsw(r, i, omega, theta)

    # Input into Planetary equation
    dadt,dhdt,dkdt,dqdt,dpdt,dLdt = GaussPlanetEqn(a,h,k,p,q,L, p_vec)

    # Simple way to advect changes into the orbital elements
    a1 = a+dadt*dt
    h1 = h+dhdt*dt
    k1 = k+dkdt*dt
    q1 = q+dqdt*dt
    p1 = p+dpdt*dt
    L1 = L+dLdt*dt

    # h1 = h+dhdt*0
    # e1 = e+dedt*0
    # theta1 = theta+dthedt*0
    # Omega1 = Omega+dOmedt*0
    # i1 = i+didt*0
    # omega1 = omega+domedt*0

    # a1 = h1**2/mu/(1-e1**2) # Get new semimajor axis from new angular momentum.

    aeioom = ahkpql2aeioom([a1,h1,k1,p1,q1,L1])

    M1 = aeioom[5] # Return the true anomaly to mean anomaly.

    # Calculate new mean anomaly
    # M1 = M_pert + mu**2/h**3*(1-e**2)**(3/2.0)*dt*rad2deg

    # Reset M1 when necessary to limit it in 360 degree.
    theta1 = M2theta(M1,h1,k1)
    if M1>360:
        M1-=360
    lambds.append(np.mod(L1-theta1+M1,360))

    # Append result to the main data array
    data.append(aeioom)
    data1.append([a1,h1,k1,p1,q1,L1])

data = np.array(data)
data1 = np.array(data1)

if plotelems == 0:
    fig,axs = plt.subplots(5,1,figsize=(13,5*5))

    ax = axs[0]
    ax.plot(ts/3600/24,data[:,0],c="green")
    ax.set_title("Semimajor Axis a")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("a [km]")

    ax = axs[1]
    ax.plot(ts/3600/24,data[:,1],c="green")
    ax.set_title("Eccentricity e")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("e")

    ax = axs[2]
    ax.plot(ts/3600/24,data[:,2],c="green")
    ax.set_title("Inclination i")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("i [deg]")

    if i_0==0:
        ax = axs[3]
        ax.plot(ts/3600/24,lambds,c="green")
        ax.set_title("$\lambda$ [deg]")
        ax.grid()
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("$\lambda$ [deg]")

    else:
        ax = axs[3]
        ax.plot(ts/3600/24,data[:,3],c="green")
        ax.set_title("Argument of periapsis $\omega$")
        ax.grid()
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("$\omega$ [deg]")

        ax = axs[4]
        ax.plot(ts/3600/24,data[:,4],c="green")
        ax.set_title("Node angle $\Omega$")
        ax.grid()
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("$\Omega$ [deg]")

elif plotelems == 1:
    fig, axs = plt.subplots(6, 1, figsize=(13, 5 * 5))

    ax = axs[0]
    ax.plot(ts / 3600 / 24, data1[:, 0], c="green")
    ax.set_title("Semimajor Axis a")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("a [km]")

    ax = axs[1]
    ax.plot(ts / 3600 / 24, data1[:, 1], c="green")
    ax.set_title("h")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("h")

    ax = axs[2]
    ax.plot(ts / 3600 / 24, data1[:, 2], c="green")
    ax.set_title("k")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("k")

    ax = axs[3]
    ax.plot(ts / 3600 / 24, data1[:, 3], c="green")
    ax.set_title("p")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("p")

    ax = axs[4]
    ax.plot(ts / 3600 / 24, data1[:, 4], c="green")
    ax.set_title("q")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("q")

    ax = axs[5]
    ax.plot(ts / 3600 / 24, lambds, c="green")
    ax.set_title("$\lambda$ [deg]")
    ax.grid()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("$\lambda$ [deg]")
plt.show()
# print(data[:,5])
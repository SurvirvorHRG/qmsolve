from tvtk.util import ctf
import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization,s,kg,meters,Hz,hbar,J,seconds,milliseconds,nm
from scipy.special import ellipj
from scipy.constants import epsilon_0
from scipy.special import mathieu_cem
from scipy.integrate import quad
from scipy.integrate import nquad
import math
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import animation
from scipy.special import gamma
# Define parameters
#♠hbar=1.054571596e-34
clight=299792458.0
echarge=1.602176462e-19
emass=9.10938188e-31
pmass=1.67262158e-27
uaumass=1.66053873e-27
epsilon0=1.0e7/(4*np.pi*clight*clight)
kBoltzmann=1.3806503e-23
N = 1
mass = 86.909
mass = mass*uaumass * kg
#mass = 1

w_o = 4e-6 * meters
Er = 7.16e-32 * J
print('Er =', Er)
epsilon = 4.1*Er
Delta = -100e13 * Hz
l = 1
P = 35e-3 * (J / seconds)

lambda_ = 4.65e-7 * meters
k = 2*(np.pi)/lambda_

omega_rho = np.sqrt((8 * epsilon) / (mass * w_o**2))
print('omega_rho =', omega_rho)
omega_nu = np.sqrt((2 * epsilon * k**2) / mass)
print('omega_nu =', omega_nu)

z_R = (np.pi*(w_o**2))/lambda_
print('z_R=', z_R)

alpha = ((np.abs(l)**2)*(lambda_**4))/(4*((np.pi)**4)*w_o**4)
q = ((np.abs(l)**2)/(4*(alpha**(3/2))))*(epsilon/Er)

print('alpha = ', alpha)
print('q =', q)

rabi = (2*np.pi)*10 * 1e6
rabi = rabi * Hz

#g = 9.8065 *   # Example value for gravity
#g = 1
g = (9.8065*meters)/(s*s)  # Example value for gravity
g1 = g
#g= 0
p = 0

U0 = 0.5 * mass * omega_rho**2
U1 = 0.5 * mass * omega_nu**2
C_p_l = np.sqrt(math.factorial(p) / (np.abs(l) + p))
a_s = 5.2383 * nm
g3d = 4*N*np.pi*hbar**2*a_s / mass 
alpha_ = 2*l
beta = 2*l
eta = 1/2 + 1/beta + 2/alpha_
muq = gamma(eta + 3/2)/gamma(1  + 2/alpha)/gamma(1 + 1/beta)*(g3d * U0**(2/alpha_) * U1**(1/beta) / 4*np.pi )
muq = muq**(2/(2*eta + 1))

def w_xi(xi):
    return w_o * np.sqrt(alpha * xi**2 + 1)

def get_U(rho,xi):
    return C_p_l**2 * w_o**2 / (w_xi(xi)**2) * (np.sqrt(2) / w_xi(xi) * ( np.sqrt(np.abs(l)/2) *w_xi(xi) + rho))**(2*l)* np.exp(-2* ( np.sqrt(np.abs(l)/2) *w_xi(xi) + rho)**2 / (w_xi(xi)**2))
    

def w(z):
    return w_o * np.sqrt(1 + (z/z_R)**2)

def potential(particle):
    phi = np.arctan2(particle.y, particle.x)
    z = particle.z
    nu = (l*phi)/k + z
    rho = np.sqrt(particle.x**2 + particle.y**2) - np.sqrt(np.abs(l) / 2) * w(particle.z)
    xi = z / (np.abs(l)/k)
    U_2_l = get_U(rho,xi)
    V = hbar * rabi**2 / Delta * U_2_l**2 * np.cos(k*nu)**2
    return V
    



def gravity_potential(particle,params):
    V = g1*mass*particle.z
    return V


Nx = 64
Ny = Nx
Nz = 128
#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=gravity_potential,
                spatial_ndim=3, N=Nx,Nz = Nz,extent=4*w_o,z_extent = 4*lambda_)


muq = 0
def psi_0(particle):
    V = potential(particle)
    psi = np.zeros_like(particle.x)
    
    indices = muq > V
    
    psi[indices] = np.sqrt((muq - V[indices])/g3d)
    """
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if muq > V[i,j,k]:
                    psi[i,j,k] = np.sqrt((muq - V[i,j,k])/g3d)
    """    
    return psi

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 2e-3 * seconds
sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
#sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step-cupy")
dt = total_time / 100000.
store_steps = 128
dt =  total_time
store_steps = 1
sim.run(psi_0, total_time = total_time,dt = dt, store_steps = store_steps,g=0)


    
#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#
dis = 5.2

visualization = init_visualization(sim)
visualization.plot_hot(t=0,L_norm = w_o,Z_norm = lambda_,unit=seconds * 1e-3,dis= dis)
visualization.plot_hot(t=total_time,L_norm = w_o,Z_norm = lambda_,unit=seconds * 1e-3,dis= dis)
visualization.final_plot3D_hot_x(L_norm = w_o,Z_norm = lambda_,unit=seconds * 1e-3,g=0,tmax=total_time)
visualization.final_plot3D_hot_y(L_norm = w_o,Z_norm = lambda_,unit=seconds * 1e-3,g=0,tmax=total_time)
visualization.final_plot3D_hot_z(L_norm = w_o,Z_norm = lambda_,unit=seconds * 1e-3,g=0,tmax=total_time)
#for i in range(129):
    #visualization.plot2D_hot_xz(t=i * total_time/128,L_norm = w_o,Z_norm = lambda_,unit=seconds * 1e-3)

for i in range(21):
    visualization.plot_hot(t=i * total_time/20,L_norm = w_o,Z_norm = lambda_,unit=seconds * 1e-3,g=g1,dis= dis)
#visualization.plot(t = 0 ,L_norm = w_o, Z_norm = lambda_,unit =milliseconds)
#visualization.animate_hot(L_norm = w_o, Z_norm = lambda_)
#visualization.final_plot_hot(L_norm = w_o, Z_norm = lambda_)
#visualization.final_plot3D_hot()
#visualization.final_plot3D_hot(L_norm = w_o, Z_norm = lambda_)
#visualization.plot_type = 'contour'
#visualization.animate(xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = False)

#for visualizing a single frame, use plot method instead of animate:


#visualization.plot2D(t = 0,L_norm = w_o,Z_norm = lambda_,unit = nanoseconds,potential_saturation = 0.1, wavefunction_saturation = 0.8)
#visualization.plot(t = 0 ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)

#visualization.animate2D(L_norm = w_o,unit = nanoseconds, potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = True)

#for i in range(11):
   #visualization.plot_hot(t = i *total_time/10,L_norm = w_o,Z_norm = lambda_)
   #visualization.subplot2D_hot(t = i *total_time/10, L_norm = w_o, Z_norm = lambda_,unit = milliseconds)
   
#visualization.plot2D(t =  total_time,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds,potential_saturation = 0.1, wavefunction_saturation = 0.1)
"""

visualization.plot(t = 2*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 4*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 6*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 8*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 10*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
"""
#for i in range(21):
    #visualization.plot2D_hot(t = i * total_time/20,L_norm = w_o, Z_norm = lambda_,unit = 1/1e-12)
#visualization.animate(L_norm = w_o, Z_norm = lambda_,unit = nanoseconds,time = 'ns',contrast_vals=[0.1,0.15])
#visualization.final_plot_m(L_norm = w_o, Z_norm = lambda_,unit = nanoseconds,time = 'ns')
#visualization.final_plot3D_m(L_norm = w_o, Z_norm = lambda_,unit = nanoseconds,time = 'ns')
#visualization.animate(L_norm = w_o, Z_norm = lambda_,unit = nanoseconds,time = 'ns',contrast_vals=[0.1,0.15])
#visualization.plot(t = total_time ,L_norm = w_o, Z_norm = lambda_,unit =nanoseconds,contrast_vals= [0.1, 0.15])
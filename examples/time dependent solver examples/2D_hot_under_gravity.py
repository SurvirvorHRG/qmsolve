from tvtk.util import ctf
import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, nanoseconds,hbar,picoseconds,microseconds,nm,s,seconds, ms,meters, Å, J, Hz, kg, femtoseconds,picoseconds
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

def radians_to_degrees(radians):
    return radians * (180.0 / math.pi)

#g = 9.8065 *   # Example value for gravity
#g = 1
g = (9.8065*meters)/(s*s)  # Example value for gravity

#g = 1
#g = (9.8065*m)/(s*s)  # Example value for gravity


def w(z):
    return w_o * np.sqrt(1 + (z/z_R)**2)


def radial_part(x, y, z):
    rho = np.sqrt(x**2 + y**2) - np.sqrt(np.abs(l) / 2) * w(z)
    return (((mass * omega_rho) / (np.pi * hbar))**(1/4)) * np.exp(((-mass * omega_rho) / (2 * hbar)) * rho**2)


def angular_part(phi, z):
    nu = (l*phi)/k + z
    return ((mass * omega_nu) / (np.pi * hbar))**(1/4) * np.exp(((-mass * omega_nu) / (2 * hbar)) * nu**2)


def axial_part(z):
    xi = z / (np.abs(l)/k)
    sin_arg = np.arcsin(z / z_R) + np.pi / 2
    ce_val = mathieu_cem(0,sin_arg, q)[0]
    sqrt_term = 1 - alpha * xi**2
    sqrt_term = np.maximum(sqrt_term, 0)
    #if sqrt_term == 0:
    #    return np.zeros_like(ce_val)
    #else:
    #    return np.sqrt(2 / np.pi) * (1 / np.sqrt(sqrt_term))**(1/4) * ce_val
    return np.sqrt(2 / np.pi) * (1 / np.sqrt(sqrt_term))**(1/4) * ce_val


def psi_function(x, y, z):
    return radial_part(x, y, z) * angular_part(np.arctan2(y, x), z) * axial_part(z)

#interaction potential
def pot(particle):
    #print(particle.z.shape)
    #particle.z = -1/2*g*t**2

    #V = -g*mass*particle.z
    #V = -10*particle.z
    V = np.zeros_like(particle.x)
    return V

def gravity(particle):
    # On essaye aussi : avec V = mass*g*particle.z particle.z = -1/2*g*t**2
    # On essaye aussi : avec V = -mass*g*particle.z particle.z = -1/2*g*t**2
    # On essaye aussi : avec V = mass*g*particle.z particle.z = 1/2*g*t**2
    # On essaye aussi : avec V = -mass*g*particle.z particle.z = 1/2*g*t**2

    #V = -g*mass*particle.z
    #V = -particle.z
    #V = -mass*N*g*particle.z
    #V = mass*N*g*particle.z
    #V = N*1*particle.z
    #V = -g*N*particle.z
    V = -mass*g*particle.y
    #V = -N*0.1*particle.z
    #V = np.zeros_like(particle.x)
    
    
    return V


#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(),
                potential=gravity,
                spatial_ndim=2, N=150,extent=10*w_o,z_extent = 10*lambda_)

H.particle_system.x = np.linspace(-H.extent/2, H.extent/2, H.N)
H.particle_system.y = np.linspace(-H.z_extent/2, H.z_extent/2, H.N)
H.particle_system.x, H.particle_system.y = np.meshgrid(H.particle_system.x,H.particle_system.y)

H.particle_system.px = np.fft.fftshift(np.fft.fftfreq(H.N, d = H.dx)) * hbar  * 2*np.pi
H.particle_system.py = np.fft.fftshift(np.fft.fftfreq(H.N, d = H.dz)) * hbar  * 2*np.pi
H.particle_system.px, H.particle_system.py = np.meshgrid(H.particle_system.px, H.particle_system.py)
H.particle_system.p2 = (H.particle_system.px**2 + H.particle_system.py**2)



def initial_wavefunction(particle):
    #my_z = np.zeros_like(particle.x)
    return psi_function(particle.x, 0, particle.y)


#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 10e-12
sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
sim.run(initial_wavefunction, total_time = total_time,dt = (0.01e-12), store_steps = 20,non_linear_function=None)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
#for i in range(21):
    #visualization.plotSI(t = i * total_time/20,L_norm = w_o, Z_norm = lambda_)
#visualization.animate(xlim=[-2*w_o*Å,2*w_o *Å], ylim=[-2*w_o*Å,2*w_o* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = True)


#for visualizing a single frame, use plot method instead of animate:
#for i in range(101):
    #visualization.plotSI(t = i * total_time/100,L_norm = w_o,Z_norm = lambda_)
visualization.plot2(t = 0,xlim=[-2*w_o,2*w_o], ylim=[-2*lambda_,2*lambda_])
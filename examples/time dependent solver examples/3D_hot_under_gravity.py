from tvtk.util import ctf
import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, nanoseconds,microseconds,nm,s,seconds, m, Å, J, Hz, kg, hbar, femtoseconds,picoseconds
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

# Parameters in SI converted to atomic units
uaumass = 1.66053873e-27*kg
N = 1000
mass = 86.909
mass = mass*uaumass

a = 5.2383
w_o = 4e-6*m
Er = 7.16e-32*J
print('Er =', Er)
epsilon = 41.6*Er
Delta = -100e13*Hz
l = 1
P = 5e-3*(J*Hz)

lambda_ = 4.65e-7*m
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

yeetr = 9.0442161272e22

def radians_to_degrees(radians):
    return radians * (180.0 / math.pi)

g = 9.8065  # Example value for gravity
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
    ce_val = mathieu_cem(0,(sin_arg),q)[0]
    sqrt_term = 1 - alpha * xi**2
    sqrt_term = np.maximum(sqrt_term, 0)
    #if sqrt_term == 0:
    #    return np.zeros_like(ce_val)
    #else:
    #    return np.sqrt(2 / np.pi) * (1 / np.sqrt(sqrt_term))**(1/4) * ce_val
    return np.sqrt(2 / np.pi) * (1 / sqrt_term)**(1/4) * ce_val


def psi_function(x, y, z):
    return radial_part(x, y, z) * angular_part(np.arctan2(y, x), z) * axial_part(z)

#interaction potential
#conv_s_ns = 1e9
#g = g/(conv_s_ns*conv_s_ns)

#g = (g*m)/(seconds*seconds)


def gravity_potential(particle):

    V = -g*mass*particle.z
    #V = -10*particle.z
    #V = np.zeros_like(particle.x)
    return V


#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(),
                potential=gravity_potential,
                spatial_ndim=3, N=100, extent=4*w_o * Å,z_extent = 4*lambda_*Å)


def initial_wavefunction(particle):
    #my_z = np.zeros_like(particle.x)
    return psi_function(particle.x, particle.y, particle.z)


#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 35 * nanoseconds
sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
sim.run(initial_wavefunction, total_time = total_time, dt = total_time/2000., store_steps = 15)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
#visualization.plot_type = 'contour'
#visualization.animate(xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = False)

#for visualizing a single frame, use plot method instead of animate:

#visualization.plot(t = 0 * femtoseconds,xlim=[-2*w_o* Å,2*w_o* Å], ylim=[-2*w_o* Å,2*w_o* Å])

"""
visualization.plot(t = 0 ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 2*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 4*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 6*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 8*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
visualization.plot(t = 10*nanoseconds ,L_norm = w_o, Z_norm = lambda_,unit = nanoseconds)
"""

visualization.animate(L_norm = w_o, Z_norm = lambda_,unit = nanoseconds,time = 'ns',contrast_vals= [0.1, 0.15])
#visualization.plot(t = total_time ,L_norm = w_o, Z_norm = lambda_,unit =nanoseconds,contrast_vals= [0.1, 0.15])
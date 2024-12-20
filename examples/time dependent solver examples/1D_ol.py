from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, m_e
import math
import scipy.special as sc
import numpy as np

#Dark-Soliton emissions with gaussian barier in 1D in SI units


# Define parameters

hbar=1.054571596e-34
clight=299792458.0
echarge=1.602176462e-19
emass=9.10938188e-31
pmass=1.67262158e-27
uaumass=1.66053873e-27
epsilon0=1.0e7/(4*np.pi*clight*clight)
kBoltzmann=1.3806503e-23
conv_C12_au=uaumass/emass


a_0 = 4 * np.pi * epsilon0 * hbar**2 / echarge / echarge / emass


# Define parameters
#mass=7.016004 * uaumass # Lithium
mass=86.909  # Atoms mass Cs 132.905 , Rb 86.909 (united atomic unit of mass)
mass  = mass * uaumass
Ntot= 20e4
omega_rho = 2*np.pi*160
omega_z = 2*np.pi*6.8


print('omega_rho =', omega_rho)

print('omega_z =', omega_z)

#V0 = 643.83 * hbar * omega_z
V0 = 1e3 * hbar * omega_z
L_r = np.sqrt(hbar/mass/omega_rho) 
L_z = np.sqrt(hbar/mass/omega_z)
sigma = 0.632 * np.sqrt(2) *L_z

#dimension-less variables
#omega_z = 0.01* omega_rho
#a_s = L * V0_tilde * np.sqrt(np.pi) / 2 / N
a_s = 94.7*a_0
g3d = Ntot * 4 *np.pi * hbar**2 * a_s / mass
#g_s = 2 * Ntot * omega_rho * a_s / omega_z  / L_z
#g_s = 2 * Ntot * omega_rho * a_s / omega_z  / L_z
#g_s = g_s * hbar
#g_s = 500 * hbar * omega_z * L_z
g_s = g3d/ 2 / np.pi / L_r / L_r
lambda_ = 790e-9

Nx = 1064                        # Grid points
Ny = Nx
tmax = 20 / omega_z               # End of propagation
dt = 0.0001 / omega_z               # Evolution step
xmax =  30 * L_z                   # x-window size
ymax = xmax                    # y-window size
images = 400                # number of .png images
absorb_coeff = 0        # 0 = periodic boundary

#dt = tmax
#images = 1

#◙muq = 0.5 * (3/2)**(2/3) * g_s**(2/3)

k = 0.5
U1 = k * mass * omega_z**2
l=1
#muq = 0.5 * (3/2)**(2/3) * g_s**(2/3)
beta = 2*l
muq = ((beta + 1)*g_s/2/beta)**(beta/(1+beta)) *U1**(1/(beta+1))
#muq = (9/4 * k * mass * omega_z**2  *g_s**2)**(1/3)
#params = 0.5


def potential(x,y,k):
    V_h = U1 * x**(beta)
    V = V_h
    return V
    
def potential_p(particle,params):
    #k = params[0]
    V_h = U1 * particle.x**(beta)
    V = V_h
    return V


def psi_0(particle,params):
    #k = params[0]
    V = potential(particle.x,0,0)

    psi = np.zeros_like(particle.x)
    for i in range(len(particle.x)):
        if muq > V[i]:
            psi[i] = np.sqrt((muq - V[i])/g_s)
        else:
            psi[i] = 0
            
    return psi



def non_linear_f2(psi,t,particle):
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    #V_h = k * mass * omega_z**2 * particle.x**2
    V_ol = 0.5 * V0 * np.cos(4*np.pi/lambda_ * particle.x)
    
    if t  < (3 / omega_z):
        V = g_s*np.abs(psi)**2 
    else:
        V = V_ol + g_s*np.abs(psi)**2 
    
    return V;

H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=potential_p,
                spatial_ndim=1, N=Nx,extent=xmax * 2,params = [U1])

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
sim.method.split_step._hbar = hbar
sim.method.split_step.set_nonlinear_term(non_linear_f2)
sim.run(psi_0, total_time =tmax, dt = dt, store_steps = images,non_linear_function=None,norm = False)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.plot1D(t = 0)
#5visualization.animate(save_animation=True)
#visualization.final_plot(L_norm = 1/L_z * 1e-3,Z_norm = 1/L_z * 1e-3,unit = omega_z*1e-3)
#visualization.save('test.txt')
visualization.plot1D(t = tmax)
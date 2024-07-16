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
omega_rho = 2*np.pi*6.8
omega_z = 2*np.pi*160



print('omega_rho =', omega_rho)

print('omega_z =', omega_z)

V0 = 100 * hbar * omega_rho
#♣V0 = 100 * hbar * omega_z
L_r = np.sqrt(hbar/mass/omega_rho) 
L_z = np.sqrt(hbar/mass/omega_z)
sigma = 0.632 * np.sqrt(2) *L_r
#V0 = 1
#sigma = 1
#dimension-less variables
#omega_z = 0.01* omega_rho
#a_s = L * V0_tilde * np.sqrt(np.pi) / 2 / N
a_s = 94.7*a_0
#g_s = 2 * Ntot * omega_rho * a_s / omega_z  / L_z
#g_s = 2 * Ntot * omega_rho * a_s / omega_z  / L_z
#g_s = g_s * hbar
#g_s = 500 * hbar * omega_z * L_z
g3d = Ntot * 4 * np.pi * hbar**2 * a_s / mass
g_s = g3d / np.sqrt(2*np.pi) / L_z

Nx = 512                        # Grid points
Ny = Nx
tmax = 20 / omega_rho               # End of propagation
dt = 0.0001 / omega_rho               # Evolution step
xmax =  30 * L_r                   # x-window size
ymax = xmax                    # y-window size
images = 400                # number of .png images


muq = (mass * omega_rho**2 *g_s / np.pi)**(1/2)

def potential(x,y):
    rho = np.sqrt(x**2 + y**2)
    V_h = 0.5 * mass * omega_rho**2 * rho**2
    #V_b = V0*np.exp(-2*(rho/sigma)**2)
    V_b = V0*np.exp(-2*(x/sigma)**2) + V0*np.exp(-2*(y/sigma)**2)
    #V_b =0
    V = V_h + V_b
    return V
    
def potential_p(particle):
    rho = particle.x**2 + particle.y**2
    V_h = 0.5 * mass * omega_rho**2 * rho
    V = V_h
    return V


def psi_0(particle):
    V = potential(particle.x,particle.y)
    psi = np.zeros_like(particle.x)
    for i in range(Nx):
        for j in range(Nx):
            if muq > V[i,j]:
                psi[i,j] = np.sqrt((muq - V[i,j])/g_s)
            else:
                psi[i,j] = 0
    return psi

def V_b(rho):
    #V_h = 0.5 * mass * omega_rho**2 * rho
    V_b = V0*np.exp(-2*(rho/sigma)**2)
    return V_b

def non_linear_f2(psi,t,particle):
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    #rho = particle.x**2 + particle.y**2
    #V_h = 0.5 * mass * omega_z**2 * particle.x**2 + particle.y**2
    #V_b = V0*np.exp(-2*((particle.x**2 + particle.y**2)/sigma)**2)
    import cupy as cp
    V_b = cp.array(V0*np.exp(-2*(particle.x/sigma)**2)) + cp.array(V0*np.exp(-2*(particle.y/sigma)**2))
    
    if t  < (3 / omega_z):
        V = V_b + g_s*abs(psi)**2 
    else:
        V = g_s*abs(psi)**2 
    
    return V;

H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=potential_p,
                spatial_ndim=2, N=Nx,extent=xmax * 2)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step-cupy")
sim.method.split_step._hbar = hbar
sim.method.split_step.set_nonlinear_term(non_linear_f2)
sim.run(psi_0, total_time =tmax, dt = dt, store_steps = 100,non_linear_function=None,norm = False)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
for i in range(101):
    visualization.plotSI(t = i * tmax/100)
#5visualization.animate(save_animation=True)
#visualization.final_plot(L_norm = 1e-3,Z_norm = 1e-3,unit = 1e-3)
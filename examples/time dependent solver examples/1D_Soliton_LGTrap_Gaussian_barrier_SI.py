# Initially, a BEC is tighly confined along the radial direction within an Laguerre-Gaussian trap 
#and more elongated along the axial direction alowing a 1D-reduction of the GPE equation. 
#separated by a Gaussian barrier in its center resulting in 2 BECs 


#At a given time, a the Guassian barrier is removed. Then,
# the two separated BECs collide.
# This results in the formation of a bunch of par dark-solitons that interacts in the trap.

#References : 
#Jameel Hussain, Javed Akram, Farhan Saif , 
#Gray/dark soliton behavior and population under a symmetric and asymmetric potential trap,
#J. Low Temp. Phy. 195, 429 (2019)




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
#mass = 22.9
#mass = 
mass=86.909  # Atoms mass Cs 132.905 , Rb 86.909 (united atomic unit of mass)
mass  = mass * uaumass
Ntot= 20e4
omega_rho = 2*np.pi*160
omega_z = 2*np.pi*6.8


print('omega_rho =', omega_rho)

print('omega_z =', omega_z)

#V0 = 643.83 * hbar * omega_z
V0 = 100 * hbar * omega_z
L_r = np.sqrt(hbar/mass/omega_rho) 
L_z = np.sqrt(hbar/mass/omega_z)
sigma = 0.632 * np.sqrt(2) *L_z

#dimension-less variables
#omega_z = 0.01* omega_rho
#a_s = L * V0_tilde * np.sqrt(np.pi) / 2 / N
a_s = 94.7*a_0
#a_s = 2.65e-9
#g_s = 2 * Ntot * omega_rho * a_s / omega_z  / L_z
g_s = 600 * hbar * omega_z * L_z
#g_s = 2 * Ntot * a_s * hbar * omega_rho
#g_s = g_s * hbar
#g_s = 500 * hbar * omega_z * L_z

Nx = 1024                        # Grid points
Ny = Nx
tmax = 20 / omega_z               # End of propagation
dt = 0.0001 / omega_z               # Evolution step
xmax =  15 * L_z                   # x-window size
ymax = xmax                    # y-window size
images = 400                # number of .png images
absorb_coeff = 0        # 0 = periodic boundary

#(dt = tmax
#images = 1


k = 0.5
#k=1.6e17
#k = 3.4e43
U1 = k * mass * omega_z**2
l=1
beta = 2*l
muq = ((beta + 1)*g_s/2/beta)**(beta/(1+beta)) *U1**(1/(beta+1))



def potential(x,y,k):
    V_h = U1 * x**(beta)
    V_b = V0*np.exp(-2*(x/sigma)**2)
    V = V_h + V_b
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
    # The linear part of the potential is a gaussian barrier with a LG trap
    #V_h = k * mass * omega_z**2 * particle.x**2
    V_b = V0*np.exp(-2*(particle.x/sigma)**2)
    
    if t  < (3 / omega_z):
        V = V_b + g_s*np.abs(psi)**2 
    else:
        V = g_s*np.abs(psi)**2 
    
    return V;

H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=potential_p,
                spatial_ndim=1, N=Nx,extent=xmax * 2,params = [U1])

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#set the time dependent simulation
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
visualization.final_plot(L_norm =1e-3,Z_norm =   1e-3,unit = 1e-3)
#visualization.save('l=1_bb.txt')

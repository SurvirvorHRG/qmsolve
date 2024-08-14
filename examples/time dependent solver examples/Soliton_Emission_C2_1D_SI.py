from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization,m_e
import math
import scipy.special as sc
import numpy as np

#Dark-Soliton emissions with gaussian barier in 1D


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
#○mass=7.016004 * uaumass # Lithium
#mass=86.909 *uaumass  # Atoms mass Cs 132.905 , Rb 86.909 (united atomic unit of mass)
mass = 1 * uaumass
#mass = 22.9 * uaumass
Ntot= 5e4
omega_rho = 1e3
omega_z = 0.01 * omega_rho

r_t = np.sqrt(hbar/mass/omega_rho) # 3e-6 meters
z_t = np.sqrt(hbar/mass/omega_z)
print('r_t =', r_t)

print('omega_rho =', omega_rho)

print('omega_z =', omega_z)

#V0 = 643.83 * hbar * omega_z

L_r = np.sqrt(hbar/mass/omega_rho) 
L_z = np.sqrt(hbar/mass/omega_z)
V0 = hbar * omega_z / 4
L = 10 * L_z
L_tilde = L/L_z/np.sqrt(2)
V0_tilde = V0 / hbar / omega_z
sigma = L / np.sqrt(2) / L_z
V0_z = V0 / hbar / omega_z
#dimension-less variables
#omega_z = 0.01* omega_rho
a_s = L * V0_tilde * np.sqrt(np.pi) / 2 / Ntot
#a_s = 94.7*a_0
g_s = 2 * Ntot * a_s * hbar * omega_rho

Nx = 1024                        # Grid points
Ny = Nx
tmax = 20 / omega_z               # End of propagation
dt = 0.0001    / omega_z            # Evolution step
xmax = 20   * L_z                # x-window size
ymax = xmax                    # y-window size
images = 1024                # number of .png images
absorb_coeff = 0        # 0 = periodic boundary
k = 0.5e35
#dt = tmax
#images = 1


U1 = k * mass * omega_z**2
l=6
beta = 2*l
muq = ((beta + 1)*g_s/2/beta)**(beta/(1+beta)) *U1**(1/(beta+1))



def potential(x,y):
    V_h = U1* x**(beta)
    #V_b = V0_tilde*( 1 - np.exp(-(x/sigma)**2) )
    V_b = 0
    V = V_h + V_b
    return V
    
    
def psi_0(particle,params):
    V = potential(particle.x,0)
    psi = np.zeros_like(particle.x)
    for i in range(len(particle.x)):
        if muq > V[i]:
            psi[i] = np.sqrt((muq - V[i])/g_s)
        else:
            psi[i] = 0
            
    return psi

def V(particle,params):        
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    V_h = U1 * particle.x**(beta)
    #V_b = V0_tilde*( 1 - np.exp(-(particle.x/sigma)**2) )
    V_b = 0
    V = V_h + V_b
    
    return V



def non_linear_f(psi,t,particle):
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    #V_b = V0*np.exp(-(particle.x/sigma)**2)
    #a1 = 1.5e-9
    #a2 = -0.2e-9
    
    a1 = 1.5e-9
    a2 = -0.2e-9
    
    if t  < 2 / omega_z:
        g1 = 2 * Ntot * a1 * hbar * omega_rho
        V = g1*np.abs(psi)**2 
    else:
        g2 = 2 * Ntot * a2 * hbar * omega_rho
        V = g2*np.abs(psi)**2 
    
    return V;

H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=V,
                spatial_ndim=1, N=Nx,extent=xmax * 2)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
sim.method.split_step._hbar = hbar
#sim.method.split_step.normalize_at_each_step(True)
sim.method.split_step.set_nonlinear_term(non_linear_f)
sim.run(psi_0, total_time =tmax, dt = dt, store_steps = images,non_linear_function=None,norm = False)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
#visualization.plot1D(t = 0)
#5visualization.animate(save_animation=True)
visualization.plot1D(t = 0)
visualization.final_plot(L_norm = 1e-6,Z_norm = 1e-6,unit = 1e-3,fixmaximum = 0.1)
#visualization.save('k=1.txt')





"""
###################################################################################
#Subplots

import matplotlib.pyplot as plt
import numpy as np

font = 44

# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(50, 15))
# Plot on each subplot using the modified final_plot function
visualization.final_plot2(axs[0],L_norm = 1/L_z * 1e-6,Z_norm = 1/L_z * 1e-6,unit = omega_z * 1e-3,fixmaximum = 0.1)
axs[0].set_title('$(a)\ U_{z} = 0.25$',fontsize = font)


# K = 0.5
k = 0.5
#muq = 0.5 * (3/2)**(2/3) * g_s**(2/3)

#muq = (9/4 * k *g_s**2)**(1/3)
U1 = k
muq = ((beta + 1)*g_s/2/beta)**(beta/(1+beta)) *U1**(1/(beta+1))
def potential2(x,y):
    V_h = k * x**2
    #V_b = V0_tilde*( 1 - np.exp(-(x/sigma)**2) )
    V_b = 0
    V = V_h + V_b
    return V
    
    
def psi_2(particle,params):
    V = potential2(particle.x,0)
    psi = np.zeros_like(particle.x)
    for i in range(len(particle.x)):
        if muq > V[i]:
            psi[i] = np.sqrt((muq - V[i])/g_s)
        else:
            psi[i] = 0
            
    return psi

def V2(particle,params):        
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    V_h = k * particle.x**2
    #V_b = V0_tilde*( 1 - np.exp(-(particle.x/sigma)**2) )
    V_b = 0
    V = V_h + V_b
    return V
    
    
H = Hamiltonian(particles=SingleParticle(m = m_e),
                potential=V2,
                spatial_ndim=1, N=Nx,extent=xmax * 2)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
#sim.method.split_step.normalize_at_each_step(True)
sim.method.split_step.set_nonlinear_term(non_linear_f)
sim.run(psi_2, total_time =tmax, dt = dt, store_steps = images,non_linear_function=None,norm = False)


visualization = init_visualization(sim)
visualization.final_plot2(axs[1],L_norm = 1/L_z * 1e-6,Z_norm = 1/L_z * 1e-6,unit = omega_z * 1e-3,fixmaximum = 0.1)
axs[1].set_title('$(b)\ U_{z} = 0.5$',fontsize = font)

# K = 0.75
k2 = 0.75
U2 = k2
muq5 = ((beta + 1)*g_s/2/beta)**(beta/(1+beta)) *U2**(1/(beta+1))
def potential3(x,y):
    V_h = 0.75 * x**2
    V_b = 0
    V = V_h + V_b
    return V
    
    
def psi_3(particle,params):
    V = potential3(particle.x,0)
    psi = np.zeros_like(particle.x)
    for i in range(len(particle.x)):
        if muq5 > V[i]:
            psi[i] = np.sqrt((muq5 - V[i])/g_s)
        else:
            psi[i] = 0
            
    return psi

def V3(particle,params):        
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    V_h = 0.75 * particle.x**2
    V_b = 0
    V = V_h + V_b
    return V
    
    
H = Hamiltonian(particles=SingleParticle(m = m_e),
                potential=V3,
                spatial_ndim=1, N=Nx,extent=xmax * 2)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
#sim.method.split_step.normalize_at_each_step(True)
sim.method.split_step.set_nonlinear_term(non_linear_f)
sim.run(psi_3, total_time =tmax, dt = dt, store_steps = images,non_linear_function=None,norm = False)


visualization = init_visualization(sim)
visualization.final_plot2(axs[2],L_norm = 1/L_z * 1e-6,Z_norm = 1/L_z * 1e-6,unit = omega_z * 1e-3,fixmaximum = 0.1)
axs[2].set_title('$(c)\ U_{z} = 0.75$',fontsize = font)


# K = 1
k = 1
#muq = (9/4 * k *g_s**2)**(1/3)
U1 = k
muq = ((beta + 1)*g_s/2/beta)**(beta/(1+beta)) *U1**(1/(beta+1))
#muq = (9/8 * mass * omega_z**2 * Ntot**2 *g_s**2)**(1/3)

def potential4(x,y):
    V_h = k * x**2
    #V_b = V0_tilde*( 1 - np.exp(-(x/sigma)**2) )
    V_b = 0
    V = V_h + V_b
    return V
    
    
def psi_4(particle,params):
    V = potential4(particle.x,0)
    psi = np.zeros_like(particle.x)
    for i in range(len(particle.x)):
        if muq > V[i]:
            psi[i] = np.sqrt((muq - V[i])/g_s)
        else:
            psi[i] = 0
            
    return psi

def V4(particle,params):        
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    V_h = k * particle.x**2
    #V_b = V0_tilde*( 1 - np.exp(-(particle.x/sigma)**2) )
    V_b = 0
    V = V_h + V_b
    return V
    
    
H = Hamiltonian(particles=SingleParticle(m = m_e),
                potential=V4,
                spatial_ndim=1, N=Nx,extent=xmax * 2)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
#sim.method.split_step.normalize_at_each_step(True)
sim.method.split_step.set_nonlinear_term(non_linear_f)
sim.run(psi_4, total_time =tmax, dt = dt, store_steps = images,non_linear_function=None,norm = False)


visualization = init_visualization(sim)
cont_rubidium = visualization.final_plot2(axs[3],L_norm = 1/L_z * 1e-6,Z_norm = 1/L_z * 1e-6,unit = omega_z * 1e-3,fixmaximum = 0.1)
axs[3].set_title('$(d)\ U_{z} = 1$',fontsize = font)

# Adjust layout
plt.tight_layout()


# Add a single colorbar for all subplots
cbar = fig.colorbar(cont_rubidium, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label('$|\psi|^2$', fontsize=44)
cbar.ax.tick_params(labelsize=40)  # Increase colorbar tick label size


# Display the plot
plt.show()

"""
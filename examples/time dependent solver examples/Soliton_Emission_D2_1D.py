from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization,NonlinearSplitStepMethod,m_e
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
#mass=7.016004 * uaumass # Lithium
mass=86.909  # Atoms mass Cs 132.905 , Rb 86.909 (united atomic unit of mass)
mass  = mass * uaumass
Ntot= 20e4
omega_rho = 2*np.pi*160
omega_z = 2*np.pi*6.8

r_t = np.sqrt(hbar/mass/omega_rho) # 3e-6 meters
z_t = np.sqrt(hbar/mass/omega_z) # 3e-6 meters
print('r_t =', r_t)

print('omega_rho =', omega_rho)

print('omega_z =', omega_z)

#V0 = 643.83 * hbar * omega_z
V0 = 100
L_r = np.sqrt(hbar/mass/omega_rho) 
L_z = np.sqrt(hbar/mass/omega_z)
sigma = 0.632

#dimension-less variables
#omega_z = 0.01* omega_rho
#a_s = L * V0_tilde * np.sqrt(np.pi) / 2 / N
a_s = 94.7*a_0
#g_s = 2 * Ntot * omega_rho * a_s / omega_z  / L_z
#g_s = 2 * Ntot * omega_rho * a_s / omega_z  / L_z
#g_s = g_s * hbar
g_s = 500
l = 2
Nx = 2000                        # Grid points
Ny = Nx
tmax = 20                # End of propagation
dt = 0.00001                # Evolution step
xmax = 10                    # x-window size
ymax = xmax                    # y-window size
images = 400                # number of .png images
absorb_coeff = 0        # 0 = periodic boundary
output_choice = 1      # If 1, it plots on the screen but does not save the images
                            # If 2, it saves the images but does not plot on the screen
                            # If 3, it saves the images and plots on the screen
fixmaximum= 0            # Fixes a maximum scale of |psi|**2 for the plots. If 0, it does not fix it.

muq = 0.5 * (3/2)**(2/3) * g_s**(2/3)

#muq = (9/8 * mass * omega_z**2 * Ntot**2 *g_s**2)**(1/3)
def potential(x,y):
    V_h = 0.5 * x**(2*l)
    V_b = V0*np.exp(-(x/sigma)**2)
    V = V_h + V_b
    return V
    
    
def psi_0(particle):
    V = potential(particle.x,0)
    psi = np.zeros_like(particle.x)
    for i in range(len(particle.x)):
        if muq > V[i]:
            psi[i] = np.sqrt((muq - V[i])/g_s)
        else:
            psi[i] = 0
            
    return psi

dt_0 = dt
def psi_1(particle):
    Vuu = potential(particle.x,0)

    psi_11 = psi_0(particle)

    
    
    U = NonlinearSplitStepMethod(Vuu, (H.extent,), -1.0j*dt_0)
    #U.normalize_at_each_step(True)
    U.set_timestep(-1.0j*dt_0)
    g1 = 0
    U.set_nonlinear_term(lambda psi,t,particle: 
                         g1*np.abs(psi)**2)

    import time
    import progressbar
    store_steps = 20
    t0 = time.time()
    bar = progressbar.ProgressBar()
    for i in bar(range(store_steps)):
        psi_11 = U(psi_11,0,particle)
    print("Took", time.time() - t0)
    return psi_11

def V(particle):        
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    V_h = 0.5 * particle.x**(2*l)
    V_b = V0*np.exp(-(particle.x/sigma)**2)
    
    V = V_h
    
    return V

def non_linear_f(psi,t,particle):
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    #print(t)
    #V_h = particle.x**2/2
    V_b = V0*np.exp(-(particle.x/sigma)**2)
    
    if t  < 3:
        V = V_b + g_s*abs(psi)**2 
    else:
        V = g_s*abs(psi)**2 
    
    return V;


def non_linear_f2(psi,t,particle):
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    V_b = V0*np.exp(-(particle.x/sigma)**2)
    
    if t  < 3:
        V = V_b + g_s*np.abs(psi)**2 
    else:
        V = g_s*np.abs(psi)**2 
    
    return V;

H = Hamiltonian(particles=SingleParticle(m = m_e),
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
sim.method.split_step.set_nonlinear_term(non_linear_f2)
#sim.method.split_step.normalize_at_each_step(True)
sim.run(psi_1, total_time =tmax, dt = dt, store_steps = images,non_linear_function=None,norm = False)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.plot1D(t = 0)
visualization.animate(save_animation=True)
visualization.final_plot(L_norm = 1,Z_norm = 1,unit = 1)
from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, milliseconds,seconds,J, meters,m_e,kg,Hz,hbar
import math
import scipy.special as sc
import numpy as np

# Define parameters
hbar_SI=1.054571596e-34
clight=299792458.0
echarge=1.602176462e-19
emass=9.10938188e-31
pmass=1.67262158e-27
uaumass=1.66053873e-27
epsilon0=1.0e7/(4*np.pi*clight*clight)
kBoltzmann=1.3806503e-23
conv_C12_au=uaumass/emass

a_0 = 4 * np.pi * epsilon0 * hbar_SI**2 / echarge / echarge / emass

# Define parameters
#mass=7.016004 * uaumass # Lithium
mass=86.909  # Atoms mass Cs 132.905 , Rb 86.909 (united atomic unit of mass)
mass  = mass * uaumass * kg
Ntot= 20e4
omega_rho = 2*np.pi*160 * Hz
omega_z = 2*np.pi*6.8 * Hz
P = 1  * J / seconds #Watt
Wox = 1.1e-6 * meters
Woz = 3.2e-6 * meters
print('omega_rho =', omega_rho)

print('omega_z =', omega_z)

V0 = 100 * hbar * omega_z
V1 = 0
L_r = 0.84e-6 * meters
L_z = 4.12e-6 * meters
sigma = 0.632
U = 2*V0*P/ np.pi / Wox / Woz 
#dimension-less variables
#omega_z = 0.01* omega_rho
#a_s = L * V0_tilde * np.sqrt(np.pi) / 2 / N
a_s = 94.7*a_0 * meters

g_s = 4*Ntot*np.pi*hbar**2*a_s / mass 
#g_s = 2 * N * omega_rho * a_s / omega_z  / L_z
#g_s = 500

Nx = 32                        # Grid points
Ny = Nx
Nz = 512
tmax = 20                # End of propagation
dt = 0.0001                # Evolution step
xmax = 40 * L_r                   # x-window size
ymax = xmax                    # y-window size
zmax = 40 * L_z                     # x-window size
images = 20                # number of .png images
absorb_coeff = 0        # 0 = periodic boundary

fixmaximum= 0            # Fixes a maximum scale of |psi|**2 for the plots. If 0, it does not fix it.

omega_mean = (omega_rho*omega_rho*omega_z)**(1/3)
a_oh = np.sqrt(hbar / mass / omega_mean)
muq = 0.5 * (15 * Ntot * a_s / a_oh)**(2/5) * hbar * omega_mean
#muq = 0.5 * (3/2)**(2/3) * g_s**(2/3)


def potential(x,y,z):
    V_rho = 0.5 * mass * omega_rho**2 * (x**2 + y**2)
    V_h = 0.5 * mass * omega_z**2 * z**2
    V_b = 1 * np.exp(-2*( (x/Wox)**2 + (z/Woz)**2 ) )
    V = V_rho + V_h + V_b
    return V
    

def psi_0(particle,params):
    V = potential(particle.x,particle.y,particle.z)
    psi = np.zeros_like(particle.x)
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if muq > V[i,j,k]:
                    psi[i,j,k] = np.sqrt((muq - V[i,j,k])/g_s)
            
    return psi
    
def psi_1D(particle):
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
    V_rho = 0.5 * mass * omega_rho**2 * (particle.x**2 + particle.y**2)
    V_h = 0.5 * mass * omega_z**2 * particle.z**2
    #V_b = U * np.exp(-2*( (particle.x/Wox)**2 + (particle.z/Woz)**2 ) )
    
    V = V_rho + V_h
    
    return V

def non_linear_f(psi,t,particle):
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    #print(t)
    V_h = particle.x**2/2
    V_b = V0*np.exp(-(particle.x/sigma)**2)
    V_1 = V1*particle.x*np.heaviside(particle.x,1)
    
    if t  < 3:
        V = V_b + V_1 + g_s*abs(psi)**2 
    else:
        V = V_1 + g_s*abs(psi)**2 
    
    return V;

def non_linear_cupy(psi,t,particle):
    import cupy as cp
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    V_b = cp.array(U * np.exp(-2*( (particle.x/Wox)**2 + (particle.z/Woz)**2 ) ))
    
    if t  < 0.02 * seconds:
        V = V_b + g_s*cp.abs(psi)**2 
    else:
        V = g_s*cp.abs(psi)**2 
    
    return V;


H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=V,
                spatial_ndim=3, N=Nx,Nz = Nz,extent=2*xmax,z_extent = 2*zmax)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step-cupy")
sim.method.split_step.set_nonlinear_term(non_linear_cupy)

total_t = tmax / omega_z
dt_t = dt / omega_z
#dt_t = 0.01 *seconds
sim.run(psi_0, total_time = total_t, dt = dt_t, store_steps = 100,non_linear_function=None,norm = False)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.plot(t = 0)
#5visualization.animate(save_animation=True)
visualization.final_plot(L_norm = meters,Z_norm = meters,unit = milliseconds)
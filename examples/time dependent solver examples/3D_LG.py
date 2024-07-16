from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization,m_e
import math
import scipy.special as sc
import numpy as np
from scipy.special import gamma

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
l = 1
Ntot= 20e4
omega_rho = 2*np.pi*160
omega_z = 2*np.pi*6.8
U0 = 0.5 * mass * omega_rho**2
U1 = 0.5 * mass * omega_z**2
alpha = 2*l
beta = 2*l
print('omega_rho =', omega_rho)
print('omega_z =', omega_z)
print('U0 =', U0)
print('U1 =', U1)
a_p = np.sqrt(hbar/mass/omega_rho)
a_z = np.sqrt(hbar/mass/omega_z)
sigma = 0.632
a_s = 94.7*a_0
g3d = 4*Ntot*np.pi*hbar**2*a_s / mass 


Nx = 128                        # Grid points
Ny = Nx
Nz = 512
tmax = 20                # End of propagation
dt = 0.0001                # Evolution step
xmax = 40 * a_p                   # x-window size
ymax = xmax                    # y-window size
zmax = 40 * a_z                     # x-window size
images = 20                # number of .png images


eta = 1/2 + 1/beta + 2/alpha
muq = gamma(eta + 3/2)/gamma(1  + 2/alpha)/gamma(1 + 1/beta)*(g3d * U0**(2/alpha) * U1**(1/beta) / 4*np.pi )
muq = muq**(2/(2*eta + 1))




def potential(x,y,z):
    rho = np.sqrt(x**2 + y**2)
    U_rho = U0 * rho**(alpha)
    U_z = U1 * z**(beta)
    return U_rho + U_z
    

def psi_0(particle):
    V = potential(particle.x,particle.y,particle.z)
    psi = np.zeros_like(particle.x)
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if muq > V[i,j,k]:
                    psi[i,j,k] = np.sqrt((muq - V[i,j,k])/g3d)
            
    return psi
    


def V(particle):        
    rho = np.sqrt(particle.x**2 + particle.y**2)
    U_rho = U0 * rho**(alpha)
    U_z = U1 * particle.z**(beta)
    return U_rho + U_z


def interaction(psi,t,particle):
    return g3d*abs(psi)*22

def non_linear_cupy(psi,t,particle):
    import cupy as cp
    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    V_b = cp.array(1 * np.exp(-2*( (particle.x)**2 + (particle.z)**2 ) ))
    
    if t  < 0.02:
        V = V_b + g3d*cp.abs(psi)**2 
    else:
        V = g3d*cp.abs(psi)**2 
    
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
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
sim.method.split_step._hbar = hbar
sim.method.split_step.set_nonlinear_term(interaction)

total_t = 100e-3
dt_t = total_t
#dt_t = 0.01 *seconds
sim.run(psi_0, total_time = total_t, dt = dt_t, store_steps = 1,non_linear_function=None,norm = False)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.plot(t = 0)
#5visualization.animate(save_animation=True)
#visualization.final_plot(L_norm = 1,Z_norm = 1,unit = 1)
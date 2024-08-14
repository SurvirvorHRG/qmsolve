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
mass=7.016004 * uaumass # Lithium
#mass=86.909  # Atoms mass Cs 132.905 , Rb 86.909 (united atomic unit of mass)
mass  = mass * uaumass
l = 1
Ntot= 5e4
omega_rho = 1e3
omega_z = 0.01 * omega_rho
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
#a_s = 94.7*a_0



V0 = hbar * omega_z / 4
L = 10 *a_z
L_tilde = L/a_z/np.sqrt(2)
V0_tilde = V0 / hbar / omega_z
sigma = L / np.sqrt(2) / a_z
V0_z = V0/hbar/omega_z
a_s = L*V0_tilde*np.sqrt(np.pi) / 2 / Ntot
g3d = 4*Ntot*np.pi*hbar**2*a_s / mass
#g3d = 100 * hbar * omega_z * a_z * 2*np.pi * (a_p**2)


Nx = 1064                        # Grid points
tmax = 20                # End of propagation
dt = 0.0001                # Evolution step
xmax = 20 * a_z                   # x-window size

eta = 1/2 + 1/beta + 2/alpha
muq = gamma(eta + 3/2)/gamma(1  + 2/alpha)/gamma(1 + 1/beta)*(g3d * U0**(2/alpha) * U1**(1/beta) / 4*np.pi )
muq = muq**(2/(2*eta + 1))
#muq = ((beta + 1)*g3d/2/beta)**(beta/(1+beta)) *U1**(1/(beta+1))

#V0 = 500 * hbar * omega_z
#sigma = 0.632 * np.sqrt(2) * a_z

def potential(x,y,z):
    U_z = U1 *z**(beta)
    #V_z = V0 * np.exp(-2*(z/sigma)**2)
    V_b = V0_tilde*(1-np.exp(-(z/sigma)**2))
    return U_z + V_b
    

def psi_0(particle,params):
    V = potential(0,0,particle.x)
    
    psi = np.zeros_like(particle.x)
    
    for i in range(Nx):
        if muq > V[i]:
            psi[i] = np.sqrt((muq - V[i])/g3d)
            
    return psi
    


def V(particle,params):        
    U_z = U1 * particle.x**(beta)
    V_b = V0_tilde*(1-np.exp(-(particle.x/sigma)**2))
    return U_z + V_b


def interaction(psi,t,particle):
    return g3d*abs(psi)*2

def non_linear(psi,t,particle):
    V = 0
    a1 = 1.5e-9
    a2 = -0.2e-9
    if t < 2/omega_z:
        g1 = 4*Ntot*np.pi*hbar**2*a1 / mass
        V = g1*np.abs(psi)**2
    else:
        g2 = 4*Ntot*np.pi*hbar**2*a2 / mass
        V = g2*np.abs(psi)**2
        
    return V


H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=V,
                spatial_ndim=1, N=Nx,extent=2*xmax)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
sim.method.split_step._hbar = hbar
#sim.method.split_step.normalize_at_each_step(True)
sim.method.split_step.set_nonlinear_term(non_linear)

total_t = tmax / omega_z
dt_t = dt / omega_z
stored = 500
stored = 1
dt_t = total_t

sim.run(psi_0, total_time = total_t, dt = dt_t, store_steps = stored,non_linear_function=None,norm = False)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.plot1D(t = 0)
#5visualization.animate(save_animation=True)
visualization.final_plot(L_norm = 1,Z_norm = 1,unit = 1)
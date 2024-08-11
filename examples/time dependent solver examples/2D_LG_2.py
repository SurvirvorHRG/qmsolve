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

Ntot= 20e4
omega_rho = 2*np.pi*6.8
omega_z = 2*np.pi*160


print('omega_rho =', omega_rho)
print('omega_z =', omega_z)

a_p = np.sqrt(hbar/mass/omega_rho)
a_z = np.sqrt(hbar/mass/omega_z)
a_s = 94.7*a_0
g3d = 4*Ntot*np.pi*hbar**2*a_s / mass/4
#g3d = 500 * hbar * omega_rho * a_p * 2*np.pi*(a_z**2)


Nx = 1024                     # Grid points
Ny = Nx
tmax = 20                # End of propagation
dt = 0.0001                # Evolution step
xmax = 20 * a_p                  # x-window size
#xmax = 2* 1e3 * a_p

l = 1
alpha = 2*l
beta = 2*l
#### THIGHING
#k = 9.25e47
#k=3.5e18

#USE THESE COEFF
#¸k = 2e47
#êk = 0.5e19
k=0.5
U0 = k * mass * omega_rho**2
U1 = k * mass * omega_z**2
print('U0 =', U0)
print('U1 =', U1)


#eta = 1/2 + 1/beta + 2/alpha
#muq = gamma(eta + 3/2)/gamma(1  + 2/alpha)/gamma(1 + 1/beta)*(g3d * U0**(2/alpha) * U1**(1/beta) / 2*np.pi )
#muq = muq**(2/(2*eta + 1))



muq = (g3d * U0**(3/alpha) * (l+1)*(alpha+2) / 2 / np.pi / l)**(alpha/(alpha + 3))


V0 = 1000 * hbar * omega_rho 
#V0 = 1
sigma = 0.632 * np.sqrt(2) * a_p
#♦V0 = V0 / np.sqrt(2*np.pi) / sigma
#sigma =5 * np.sqrt(2) * a_p

def potential(x,y,z):
    rho = np.sqrt(x**2 + y**2)
    U_rho = U0 *rho**(alpha)
    V_rho = V0 * np.exp(-2*(rho/sigma)**2) 
    #V_rho = V0 * np.exp(-2*(x/sigma)**2) + V0 * np.exp(-2*(y/sigma)**2) 
    #V_rho = V0 * np.exp(-2*(x/sigma)**2) 
    #V_rho = V0 * np.exp(-0.5*(x/sigma)**2)
    return U_rho + V_rho
    

def psi_0(particle,params):
    V = potential(particle.x,particle.y,0)
    psi = np.zeros_like(particle.x)
    
    for i in range(Nx):
        for j in range(Nx):
            if muq > V[i,j]:
                psi[i,j] = np.sqrt((muq - V[i,j])/g3d)
            
    return psi
    


def V(particle,params): 
    rho = np.sqrt(particle.x**2 + particle.y**2)       
    U_rho = U0 * rho**(alpha)
    return U_rho



def non_linear(psi,t,particle):
    import cupy as cp
    V = 0
    rho = np.sqrt(particle.x**2 + particle.y**2)
    if t < 0.02:
        V = V0 * np.exp(-2*(rho/sigma)**2)
        #V = V0 * np.exp(-2*(particle.x/sigma)**2) + V0 * np.exp(-2*(particle.y/sigma)**2) 
        #V = V0 * np.exp(-2*(particle.x/sigma)**2)
       # V = V0 * np.exp(-0.5*(particle.x/sigma)**2)
    else:
        V = 0
        
    return cp.array(V) + g3d*abs(psi)**2


H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=V,
                spatial_ndim=2, N=Nx,extent=2*xmax)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step-cupy")
sim.method.split_step._hbar = hbar
sim.method.split_step.set_nonlinear_term(non_linear)

total_t = 0.2
dt_t = 1e-5 
stored = 1024
#/stored = 1
#dt_t = total_t

sim.run(psi_0, total_time = total_t, dt = dt_t, store_steps = stored,non_linear_function=None,norm = False,absorb_coeff=0)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
#visualization.plotSI(t = 0)

#for i in range(201):
    #visualization.plotSI(i * total_t/200)
#5visualization.animate(save_animation=True)
#visualization.final_plot_x(L_norm = 1,Z_norm = 1,unit = 1)
visualization.final_plot_3D_x(L_norm = 1e-6, Z_norm = 1e-6,unit = 1e-3,extent=xmax,fixmaximum = 0)

#☺for i in range(21):
    #visualization.plotSI(i * total_t/20)
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
l = 3
Ntot= 20e4
omega_rho = 2*np.pi*6.8
omega_z = 2*np.pi*6.8
#k= 0.5
#k = 0.5e46
k = 0.5e18
U0 = k * mass * omega_rho**2
U1 = k * mass * omega_z**2
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
g3d = 4*Ntot*np.pi*hbar**2*a_s / mass /4


Nx = 128                        # Grid points
Ny = Nx
Nz = 128
tmax = 20                # End of propagation
dt = 0.0001                # Evolution step
xmax = 25 * a_p                   # x-window size
ymax = xmax                    # y-window size
zmax = 25 * a_z                     # x-window size


eta = 1/2 + 1/beta + 2/alpha
muq = gamma(eta + 3/2)/gamma(1  + 2/alpha)/gamma(1 + 1/beta)*(g3d * U0**(2/alpha) * U1**(1/beta) / 4*np.pi )
muq = muq**(2/(2*eta + 1))

print('muq = ', muq)
V0 = 500 * hbar * omega_z
sigma = 0.5*0.632 * np.sqrt(2) * a_z


def potential(x,y,z):
    rho = np.sqrt(x**2 + y**2)
    #c =  np.sqrt(x**2 + y**2 + z**2)
    V_b = V0*np.exp(-2*(z/sigma)**2 )
    #V_b = V0*np.exp(-2*(c/sigma)**2 )
    U_rho = U0 * rho**(alpha)
    U_z = U1 * z**(beta)
    return U_rho + V_b + U_z
    

def psi_0(particle,params):
    V = potential(particle.x,particle.y,particle.z)
    psi = np.zeros_like(particle.x)
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if muq > V[i,j,k]:
                    psi[i,j,k] = np.sqrt((muq - V[i,j,k])/g3d)
            
    return psi
    


def V(particle,params):        
    rho = np.sqrt(particle.x**2 + particle.y**2)
    U_rho = U0 * rho**(alpha)
    U_z = U1 * particle.z**(beta)
    return U_rho + U_z


def interaction(psi,t,particle):
    import cupy as cp
    V = 0
    #c =  np.sqrt(particle.x**2 + particle.y**2 + particle.z**2)
    if t < 0.02:
        V =  V0*np.exp(-2*(particle.z/sigma)**2 )
        #V = V0*np.exp(-2*(c/sigma)**2 )
    else:
        V = 0
    return cp.array(V) + g3d*cp.abs(psi)*2



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
sim.method.split_step._hbar = hbar
sim.method.split_step.set_nonlinear_term(interaction)

total_t = 0.40
dt_t = 1e-5
stored = 100
#dt_t = total_t
#stored = 1
sim.run(psi_0, total_time = total_t, dt = dt_t, store_steps = stored,non_linear_function=None,norm = False)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.plot(t = 0,L_norm = 1e-3,Z_norm = 1e-3,unit = 1e-3,azimuth = 0,elev = 90,dis = 3.25)
#5visualization.animate(save_animation=True)
visualization.final_plot(L_norm = 1e-3,Z_norm = 1e-3,unit = 1e-3)

for i in range(51):
    visualization.plot(t = i * total_t/50,L_norm = 1e-3,Z_norm = 1e-3,unit = 1e-3,azimuth = 0,elev = 90,dis = 3.25)
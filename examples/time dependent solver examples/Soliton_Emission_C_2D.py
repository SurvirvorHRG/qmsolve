import numpy as np

from qmsolve import Hamiltonian, SingleParticle, TimeSimulation,NonlinearSplitStepMethod, init_visualization,meters, m_e

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

# Define parameters
#mass=86.909 * uaumass # Lithium
mass=7.016004 * uaumass # Lithium
N= 5e4
omega_rho = 1.0e3 # 1kHz
r_t = np.sqrt(hbar/mass/omega_rho) # 3e-6 meters
print('r_t =', r_t)

print('omega_rho =', omega_rho)
omega_z = 0.01 * omega_rho
print('omega_z =', omega_z)

V0 = (hbar*omega_rho)/4
L = 15*r_t

#dimension-less variables
V0_tilde = V0/(hbar*omega_rho)
L_tilde = L/r_t
omega_z = 0.01* omega_rho
a_s = L * V0_tilde * np.sqrt(np.pi) / 2 / N

Nx = 2000                        # Grid points
Ny = Nx
tmax = 100                # End of propagation
dt = tmax/5000                # Evolution step
xmax =  2 *100                    # x-window size
ymax = xmax                    # y-window size
images = 100                # number of .png images



def psi_0(particle):
    return np.sqrt(N) / np.pi**(1/4) / np.sqrt(L_tilde) * np.exp(-(particle.x/(np.sqrt(2)*L_tilde))**2)




def V(particle):        

    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    
    V_d = V0*(1-np.exp(-(particle.x/L)**2))
    V_z = 0.5  * mass* omega_z**2 * particle.x**2
    
    #V_d += V_z
    
    f_n  = ( V_d) / hbar / omega_rho
    
    V = f_n


    return V;

def interaction(psi,t,particle):
    
    a1 = 1.5e-9
    a2 = -0.2e-9
    V = 0

    if t<8:
        g1 = 2* a1 / r_t
        V= g1*abs(psi)**2 
    else:
        g2 = 2* a2 / r_t
        V= g2*abs(psi)**2
        
    return V
    

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = m_e), 
                potential = V, 
                spatial_ndim = 1, N = Nx, extent=xmax )


total_time = tmax
DT = dt = total_time/10000

dt_0 = 0.000001
def psi_1(particle):
    Vuu = V(particle)

    psi_11 = psi_0(particle)

    
    
    U = NonlinearSplitStepMethod(Vuu, (H.extent,), -1.0j*dt_0,1)
    #U.normalize_at_each_step(True)
    U.set_timestep(-1.0j*dt_0)
    g1 = 2* a_s / r_t
    U.set_nonlinear_term(lambda psi,t,particle: 
                         g1*np.abs(psi)**2)

    import time
    import progressbar
    store_steps = 100000
    t0 = time.time()
    bar = progressbar.ProgressBar()
    for i in bar(range(store_steps)):
        psi_11 = U(psi_11,0,particle)
    print("Took", time.time() - t0)
    return psi_11
#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
#total_time = tmax
#set the time dependent simulation
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
sim.method.split_step.set_nonlinear_term(interaction)
sim.run(psi_1, total_time = total_time, dt = dt, store_steps = 100,non_linear_function=None)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.final_plot(L_norm = 1,unit = 1 )

import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation,NonlinearSplitStepMethodCupy, init_visualization,ms,seconds,hbar,kg, Hz,meters,nm,m_e, Å,milliseconds

#=========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#

# File: ./examples1D/Solitons_in_phase_1D.py
# Run as    python3 bpm.py Solitons_in_phase_1D 1D
# Encounter of two solitons in phase, which traverse each other unaffected.
# During the collision, an interference pattern is generated.

# Define parameters
uaumass=1.66053873e-27
# Define parameters
mass=7.016004 * uaumass # Lithium
#mass=86.909
mass=mass*kg
m = mass
Ntot = 5e4

omega_rho = 1.0e3*Hz # 1kHz
#r_t = 3e-6*m   # 3 micro meters
r_t = np.sqrt(hbar/mass/omega_rho) # 3e-6 meters

print('r_t =', r_t)

print('omega_rho =', omega_rho)
omega_z = 0.01 * omega_rho
z_t = np.sqrt(hbar/mass/omega_z)
#omega_z = omega_rho
print('omega_z =', omega_z)

V0 = (hbar*omega_rho)/4
L = 15*r_t

#dimension-less variables
V0_tilde = V0/(hbar*omega_rho)
L_tilde = L/r_t
a_s = L*V0_tilde*np.sqrt(np.pi)/2/Ntot
N_g = 4 * np.pi * hbar**2 * a_s / m
N_g = N_g  


omega_mean = (omega_rho*omega_rho*omega_z)**(1/3)
a_oh = np.sqrt(hbar / m / omega_mean)
muq0 = 0.5 * (15 * Ntot * a_s / a_oh)**(2/5) * hbar * omega_mean

zmax = 2* 2*100* r_t       # x-window size
xmax =     zmax /8
#xmax = 2*0.3e-3 * meters





def free(particle):
    return np.zeros_like(particle.x)

def pot(particle):
    import cupy as cp
    rho = cp.array(np.sqrt(particle.x**2 + particle.y**2))
    V_z = 0.5 * mass * omega_z**2 + V0*(1 - np.exp(-(particle.z/L)**2))
    V_rho = 0.5 * mass * omega_rho**2 * rho**2   #axial trap
    V = cp.array(V_rho) + cp.array(V_z)
    return cp.array(V)

# Initial wave function (ground state of the harmonic oscillator)

def ground(x,y,z):
    rho = x**2 + y**2
    V_rho = 0.5 * m * omega_rho**2 * rho
    V_z = 0
    #V_z = 0.5 * m * omega_z**2 * z**2
    #V_z = 0.5 * m * omega_rho**2 * z**2
    #V_z = 0
    V_d = V0*(1 - np.exp(-(z/L)**2))
    #V_d = 0
    #V_d = 0
    return V_rho +V_d + V_z
    
    
def potential(particle):
    x = particle.x
    y = particle.y
    z = particle.z
    rho = x**2 + y**2
    V_rho = 0.5 * m * omega_rho**2 * rho
    #V_z = 0.5 * m * omega_rho**2 * z**2
    #V_z = 0.5 * m * omega_z**2 * z**2
    V_z = 0
    V_d = V0*(1 - np.exp(-(z/L)**2))
    #V_d = 0
    #return np.zeros_like(x)
    return V_rho + V_z + V_d
    

N = 64
Nz = 512

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = mass), 
                potential = potential, 
                spatial_ndim = 3, N = N,Nz = Nz, extent=xmax,z_extent=zmax)


def mu_f(particle):
    V = ground(particle.x,particle.y,particle.z)
    mu = ((Ntot * N_g +  np.sum(V)) * H.dx * H.dx * H.dz ) / (H.extent * H.extent * H.z_extent)
    return mu

total_time = 160e-3 * seconds
DT = dt = ( 1e-5 * seconds)
#DT = dt = total_time
stored = 100


# jjkkllmlmlmmmmmmmmmlj

dt_0 = 0.0001
def psi_1(particle):
    import cupy as cp
    Vuu = ground(particle.x,particle.y,particle.z)

    #muq = mu_f(particle)
    muq = muq0
    psi_0 = np.zeros((N,N,Nz),dtype = np.complex128)
    #psi_0 = np.exp( -(particle.x/L)**2  -(particle.y/L)**2 -(particle.z/L)**2)
    #psi_0 = ground(particle.x,particle.y,particle.z)
    
    for i in range(N):
        for j in range(N):
            for k in range(Nz):
                if muq > Vuu[i,j,k]:
                    psi_0[i,j,k] = np.sqrt((muq - Vuu[i,j,k])  / N_g )
                else:
                    psi_0[i,j,k] = 0
    
    #  print(psi_0)
    
    N_gi = 4 * np.pi * hbar**2 * a_s / m
    N_gi = N_gi  
    U = NonlinearSplitStepMethodCupy(Vuu, (H.extent, H.extent, H.z_extent), -1.0j*dt_0,mass)
    U.set_timestep(-1.0j*dt_0)
    U.set_nonlinear_term(lambda particle,t,psi:
                         N_gi*np.abs(psi)**2)
   # for i in range(20000):
    #    psi_0 = U(particle,0,psi_0).get()
    import time
    import progressbar
    store_steps = 100
    t0 = time.time()
    bar = progressbar.ProgressBar()
    for i in bar(range(store_steps)):
        psi_0 = U(0,0,psi_0).get()
    print("Took", time.time() - t0)
    return psi_0


# ThomasFermi wave-function 
def psi_0(particle):
    Vuu = ground(particle.x,particle.y,particle.z)
    #muq = mu_f(particle)
    muq = muq0
    psi_0 = np.zeros((N,N,Nz),dtype = np.complex128)
    
    for i in range(N):
        for j in range(N):
            for k in range(Nz):
                if muq > Vuu[i,j,k]:
                    psi_0[i,j,k] = np.sqrt((muq - Vuu[i,j,k]) / Ntot / N_g )
                else:
                    psi_0[i,j,k] = 0
                     
    return psi_0

# Solitons creation by changing as 
    
def interaction(psi,t,particle):
    
    a0=0;  # initial (vanishing) nonlinear coefficient    
    
    

    if t<8e-3 * seconds:
       a0 = 1.5 * nm
    else:
       a0 = -0.2 * nm
    #import cupy as cp
    #g0 = cp.array(4*np.pi*a0/mass)
    g0 = Ntot * 4*np.pi*a0/mass
    #return a0*abs(psi)**2
    return g0*abs(psi)**2




#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#

#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
#sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step-cupy")
sim.method.split_step.set_nonlinear_term(interaction)

sim.run(psi_0, total_time = total_time, dt = DT, store_steps = stored,non_linear_function=None)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
# Plot 4D of the initial wave-function 
visualization.plot(t = 0,L_norm = meters * 1e-3, Z_norm = meters * 1e-3)
# Plot of the wavefunction  in the plane (z,t)
visualization.final_plot(L_norm = meters * 1e-3, Z_norm = meters * 1e-3)
#visualization.animate(xlim=[-xmax/2 * Å,xmax/2 * Å], animation_duration = 10, save_animation = True, fps = 30)
for i in range(11):
    visualization.plot(t = i * total_time/10,L_norm = meters * 1e-3, Z_norm = meters * 1e-3)

#for visualizing a single frame, use plot method instead of animate:
#visualization.plot(t = 0 ,xlim=[-xmax* Å,xmax* Å])
#visualization.final_plot()
#visualization.plot(t = 160 * milliseconds,xlim=[-xmax* Å,xmax* Å])
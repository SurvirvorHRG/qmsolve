import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation,NonlinearSplitStepMethodCupy, init_visualization
import scipy.constants as const
#=========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#

# File: ./examples1D/Solitons_in_phase_1D.py
# Run as    python3 bpm.py Solitons_in_phase_1D 1D
# Encounter of two solitons in phase, which traverse each other unaffected.
# During the collision, an interference pattern is generated.

# Define parameters
uaumass=1.66053873e-27
hbar = const.hbar
# Define parameters
mass=7.016004 * uaumass # Lithium

mass=mass
m = mass
Ntot = 5e4

omega_rho = 1.0e3 # 1kHz
#r_t = 3e-6*m   # 3 micro meters
r_t = np.sqrt(hbar/mass/omega_rho) # 3e-6 meters
print('r_t =', r_t)

print('omega_rho =', omega_rho)
omega_z = 0.01 * omega_rho
#omega_z =  omega_rho
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
#omega_mean = (omega_rho*omega_rho*omega_rho)**(1/3)
a_oh = np.sqrt(hbar / m / omega_mean)
muq = 0.5 * (15 * Ntot * a_s / a_oh)**(2/5) * hbar * omega_mean
xmax =  0.75 * 1e-3     # x-window size
N = 128
Nz = 512



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
    V_z = 0.5 * m * omega_z**2 * z**2
    V_d = V0*(1 - np.exp(-(z/L)**2))
    return V_rho + V_z
    
    
def potential(particle):
    x = particle.x
    y = particle.y
    z = particle.z
    rho = x**2 + y**2
    V_rho = 0.5 * m * omega_rho**2 * rho
    V_d = V0*(1 - np.exp(-(z/L)**2))
    #return np.zeros_like(x)
    return V_rho  + 0.5 * m * omega_z**2 * z**2 + V_d
    
def psi_0(particle):
    Vuu = ground(particle.x,particle.y,particle.z)
    #print(Vuu)
    psi_0 = np.zeros((N,N,N),dtype = np.complex128)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if muq > Vuu[i,j,k]:
                    psi_0[i,j,k] = np.sqrt((muq - Vuu[i,j,k]) / Ntot / N_g )
                else:
                    psi_0[i,j,k] = 0
    #print(psi_0)
    return psi_0
    
def psi_1(particle):
    import cupy as cp
    Vuu = ground(particle.x,particle.y,particle.z)
    U = SplitStepMethodCupy(Vuu, (H.extent, H.extent, H.z_extent), -1.0j*DT)
    U.set_timestep(-1.0j*DT)
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
    #print(psi_0)
    
    for i in range(100):
        psi_0 = U(cp.array(psi_0)).get()
        
    U = NonlinearSplitStepMethod(Vuu, (L, ),-1.0j* DT)
    U.normalize_at_each_step(True)
    U.set_nonlinear_term(zz)
    
    for i in range(100):
        psi_1 = U(psi_1)
        
    import time
    import progressbar
    Nt = t.shape[0]
    store_steps = 100
    total_time = T
    dt_store = total_time/store_steps
    Nt_per_store_step = int(np.round(dt_store / dt))
    t0 = time.time()
    bar = progressbar.ProgressBar()
    for i in bar(range(store_steps)):
        tmp = np.copy(psi_1)
        for j in range(Nt_per_store_step):
            tmp = U(tmp)
        psi_1 = tmp
        fig = plt.figure("1D plot")    # figure
        plt.clf()                       # clears the figure
        fig.set_size_inches(8,6)
        plt.plot(X, abs(psi_1)**2)  # makes the plot
        plt.xlabel('$x$')           # format LaTeX if installed (choose axes labels, 
        plt.ylabel('$|\psi|^2$')    # title of the plot and axes range
        plt.title('$t=$ %f'%((i+j)*dt))    # title of the plot
        plt.show()
                
    print("Took", time.time() - t0)
    return psi_0


def interaction(particle,t,psi):
    a2 = -0.1e-9
    a1 = 5e-9
    a_z = a2 + (a1 -a2)*np.exp(-1*(particle.z/L)**2)
    """
    if t < 8 * milliseconds:
        a_z = 1.5 * nm
    else:
        a_z = -0.2 * nm
    """
    import cupy as cp
    g0 = cp.array(4*np.pi*a_z/mass)
    return Ntot*g0*abs(psi)**2

def zz(psi,t,particle):
    import cupy as cp
    a1 = 5e-9 
    a2 = -0.1e-9
    g_z = a2 + (a1 - a2)*np.exp(- (particle.z/L)**2)
    N_gn = 4 * np.pi * hbar**2 * cp.array(g_z) / m
    N_gn = Ntot * N_gn
    return N_gn*cp.abs(psi)**2

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = mass), 
                potential = potential, 
                spatial_ndim = 3, N = N,Nz = Nz, extent = xmax)



#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
total_time = 400e-3
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step-cupy")
sim.method.split_step.set_nonlinear_term(interaction)
sim.run(psi_1, total_time = total_time, dt = ( 1e-3), store_steps = 100)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.final_plot(unit = 1e-3)
#visualization.animate(xlim=[-xmax/2 * Å,xmax/2 * Å], animation_duration = 10, save_animation = True, fps = 30)
#for i in range(101):
    #visualization.plotSI( i * total_time / 100,L_norm = 1,Z_norm = 1)

#for visualizing a single frame, use plot method instead of animate:
#visualization.plot(t = 0 ,xlim=[-xmax* Å,xmax* Å])
#visualization.final_plot()
#visualization.plot(t = 160 * milliseconds,xlim=[-xmax* Å,xmax* Å])
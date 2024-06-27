import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization

#=========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#

# File: ./examples1D/Solitons_in_phase_1D.py
# Run as    python3 bpm.py Solitons_in_phase_1D 1D
# Encounter of two solitons in phase, which traverse each other unaffected.
# During the collision, an interference pattern is generated.

# Define parameters
hbar = 1
clight=299792458.0
echarge=1.602176462e-19
emass=9.10938188e-31
pmass=1.67262158e-27
uaumass=1.66053873e-27
epsilon0=1.0e7/(4*np.pi*clight*clight)
kBoltzmann=1.3806503e-23



aulength=4*np.pi*epsilon0*(hbar**2)/(emass*echarge*echarge)
print('aulength =',aulength )

auenergy=hbar*hbar/(emass*aulength*aulength)
print('auenergy =',auenergy )

autime=hbar/auenergy
print('autime =',autime )

conv_au_fs=autime/1.0e-15
conv_C12_au=uaumass/emass
conv_au_ang=aulength/1.0e-10
conv_K_au=kBoltzmann/auenergy


conv_J_aue = 2.2937126583579E+17


# Define parameters
mass=7.016004 * uaumass # Lithium

mass=mass
m = mass
Ntot = 3.5e4

omega_rho = 1.0e3 # 1kHz
#r_t = 3e-6*m   # 3 micro meters
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
a_s = L*V0_tilde*np.sqrt(np.pi)/2/Ntot
N_g = 4 * np.pi * hbar**2 * a_s / m
N_g = N_g  


omega_mean = (omega_rho*omega_rho*omega_z)**(1/3)
#omega_mean = (omega_rho*omega_rho*omega_rho)**(1/3)
a_oh = np.sqrt(hbar / m / omega_mean)
muq = 0.5 * (15 * Ntot * a_s / a_oh)**(2/5) * hbar * omega_mean
xmax =  0.75 * 1e-3 * 1e10      # x-window size
N = 500

def tho(t):
    return omega_rho*t

def eta(x):
    return x/r_t
def free(particle):
    return np.zeros_like(particle.x)

def pot(particle):
    import cupy as cp
    y = 0
    rho = cp.array(np.sqrt(particle.x**2 + y**2))
    V_z = 0.5 * mass * omega_z**2 + V0*(1 - np.exp(-(particle.y/L)**2))
    V_rho = 0.5 * mass * omega_rho**2 * rho**2   #axial trap
    V = cp.array(V_rho) + cp.array(V_z)
    return cp.array(V)

# Initial wave function (ground state of the harmonic oscillator)

def ground(x,y,z):
    rho = x**2 + y**2
    V_rho = 0.5 * m * omega_rho**2 * rho
    #V_z = 0.5 * m * omega_z**2 * z**2
    V_z = 0.5 * m * omega_z**2 * z**2
    V_d = V0*(1 - np.exp(-(z/L)**2))
    return V_rho + V_d  + V_z
    
    
def potential(particle):
    x = particle.x
    y = 0
    z = particle.y
    rho = x**2 + y**2
    V_rho = 0.5 * m * omega_rho**2 * rho
    V_d = V0*(1 - np.exp(-(z/L)**2))
    #return np.zeros_like(x)
    return V_rho  + 0.5 * m * omega_z**2 * z**2 + V_d
    
def psi_0(particle):
    Vuu = ground(particle.x,0,particle.y)
    psi_0 = np.zeros((N,N),dtype = np.complex128)
    for i in range(N):
        for j in range(N):
            if muq > Vuu[i,j]:
                psi_0[i,j] = np.sqrt((muq - Vuu[i,j]) / Ntot / N_g )
            else:
                psi_0[i,j] = 0
    return psi_0
    



def interaction(psi,t,particle):
    a2 = -0.1*nm
    a1 = 5 * nm
    a_z = a2 + (a1 -a2)*np.exp(-1*(particle.z/L)**2)
    """
    if t < 8 * milliseconds:
        a_z = 1.5 * nm
    else:
        a_z = -0.2 * nm
    """
    import cupy as cp
    g0 = cp.array(4*np.pi*a_z/mass)
    return N*g0*abs(psi)**2

def test(particle,t,psi):
    return gint*np.abs(psi)**2


def zz(psi,t,particle):
    import cupy as cp
    a1 = 5e-9 * meters
    a2 = -0.1e-9 * meters
    g_z = a2 + (a1 - a2)*np.exp(- (particle.y/L)**2)
    N_gn = 4 * np.pi * hbar**2 * cp.array(g_z) / m
    N_gn = Ntot * N_gn
    return N_gn*cp.abs(psi)**2

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = mass), 
                potential = potential, 
                spatial_ndim = 2, N = N, extent = xmax * Å)



#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
total_time = 160e-3 * seconds
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step-cupy")
sim.method.split_step.set_nonlinear_term(test)
sim.run(psi_0, total_time = total_time, dt = ( 1e-5 * seconds), store_steps = 100,non_linear_function=None)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.final_plot()
#visualization.animate(xlim=[-xmax/2 * Å,xmax/2 * Å], animation_duration = 10, save_animation = True, fps = 30)
#for i in range(101):
    #visualization.plotSI( i * total_time / 100,L_norm = 1,Z_norm = 1)

#for visualizing a single frame, use plot method instead of animate:
#visualization.plot(t = 0 ,xlim=[-xmax* Å,xmax* Å])
#visualization.final_plot()
#visualization.plot(t = 160 * milliseconds,xlim=[-xmax* Å,xmax* Å])
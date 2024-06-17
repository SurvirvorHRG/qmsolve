import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization,ms,seconds,hbar,kg, Hz,m,nm,m_e, Å,milliseconds

#=========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#

# File: ./examples1D/Solitons_in_phase_1D.py
# Run as    python3 bpm.py Solitons_in_phase_1D 1D
# Encounter of two solitons in phase, which traverse each other unaffected.
# During the collision, an interference pattern is generated.

# Define parameters
#hbar = 1
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

mass=mass*kg
N= 5e4

omega_rho = 1.0e3*Hz # 1kHz
#r_t = 3e-6*m   # 3 micro meters
r_t = np.sqrt(hbar/mass/omega_rho) # 3e-6 meters
print('r_t =', r_t)

print('omega_rho =', omega_rho)
omega_z = 0
print('omega_z =', omega_z)

V0 = (hbar*omega_rho)/4
L = 15*r_t

#dimension-less variables
V0_tilde = V0/(hbar*omega_rho)
L_tilde = L/r_t
omega_z = 0.01* omega_rho

xmax =  2 * 0.3e-3 * 1e10             # x-window size


def tho(t):
    return omega_rho*t

def eta(x):
    return x/r_t
def free(particle):
    return np.zeros_like(particle.x)

def pot(particle):
    import cupy as cp
    rho = cp.array(np.sqrt(particle.x**2 +particle.y**2))
    Vd = 0.5 * mass * omega_z**2 + V0*(1 - np.exp(-(particle.z/L)**2))
    Vz = 0.5 * mass * omega_rho**2 * rho**2   #axial trap
    V = cp.array(Vd) + cp.array(Vz)
    return cp.array(V)

def initial_wavefunction(particle): # Initial wavefunction ground-state: a Gaussian 
    return np.sqrt(N)/(np.pi**(1/4) * np.sqrt(L_tilde)) * np.exp( -1*eta(particle.z)**2/(2*(L_tilde**2) ) )
    
a0 = L*V0_tilde*np.sqrt(np.pi)/2/N


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
"""
    a1 = 1.5 * nm
    a2 = -0.2 * nm
    g1 = 4*np.pi*a1/mass
    g2 = 4*np.pi*a2/mass
    if t  < 8 * milliseconds :
        return N*g1*abs(psi)**2
    else:
        return N*g2*abs(psi)**2
"""
#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = mass), 
                potential = pot, 
                spatial_ndim = 3, N = 100, extent = xmax * Å)



#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
total_time = 400e-3 * seconds
#set the time dependent simulation
##sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
sim.run(initial_wavefunction, total_time = total_time, dt = ( 1e-4 * seconds), store_steps = 100,non_linear_function=interaction)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
#visualization.animate(xlim=[-xmax/2 * Å,xmax/2 * Å], animation_duration = 10, save_animation = True, fps = 30)


#for visualizing a single frame, use plot method instead of animate:
#visualization.plot(t = 0 ,xlim=[-xmax* Å,xmax* Å])
#visualization.final_plot()
#visualization.plot(t = 160 * milliseconds,xlim=[-xmax* Å,xmax* Å])
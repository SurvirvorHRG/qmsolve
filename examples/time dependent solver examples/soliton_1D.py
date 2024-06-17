import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization,J,ms,seconds,kg, Hz,nm,m_e,milliseconds,meters

#=========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#

# File: ./examples1D/Solitons_in_phase_1D.py
# Run as    python3 bpm.py Solitons_in_phase_1D 1D
# Encounter of two solitons in phase, which traverse each other unaffected.
# During the collision, an interference pattern is generated.

# Define parameters
#hbar = 1
import scipy.constants as const

# Constants and parameters
hbar = const.hbar
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
N= 5e4

omega_rho = 1.0e3 # 1kHz
#omega_rho = 1.0e3 # 1kHz
#r_t = 3e-6*m   # 3 micro meters
#r_t = np.sqrt(hbar/mass/omega_rho) # 3e-6 meters
r_t = 3e-6 # 3e-6 meters
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

#xmax =  200*(r_t/m) * 1e10  *   Å         # x-window size
xmax =  200*r_t



def tho(t):
    return omega_rho*t

def eta(x):
    return x/r_t
def free(particle):
    return np.zeros_like(particle.x)

def pot(particle):
    rho = np.sqrt(particle.x**2 +particle.y**2)
    #Vz = 0.5 * mass * omega_z**2 + V0*(1 - np.exp(-(particle.x/L)**2))
    Vz = V0*(1 - np.exp(-(particle.z/L)**2))
    Vd = 0.5 * mass * omega_rho**2 * rho**2   #axial trap
    V = Vd + Vz
    return V

def initial_wavefunction(particle): # Initial wavefunction ground-state: a Gaussian 
    return np.sqrt(N)/(np.pi**(1/4) * np.sqrt(L)) * np.exp( -1* particle.x**2/(2*(L**2) )
                                                          -1* particle.y**2/(2*(L**2) ) 
                                                          -1* particle.z**2/(2*(L**2) ))
    
a0 = L*V0_tilde*np.sqrt(np.pi)/2 / N
gint = 4 * np.pi * hbar**2 * a0 / mass
gint = N * gint


def test(particle,t,psi):
    return gint*np.abs(psi)**2


def a(x):
    a2  = 5-9
    a1 = -1e-9
    a_z = a2 + (a1 - a2)*np.exp( - (x /L)**2)
    gint = (4*np.pi*a_z/mass)
    return gint


#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = mass), 
                potential = pot, 
                spatial_ndim = 3, N = 100, extent = xmax )


DT = 1e-5 
#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
total_time = 160e-3 
#set the time dependent simulation
#sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson")
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step-cupy")
sim.method.split_step.set_nonlinear_term(test)
sim.run(initial_wavefunction, total_time = total_time, dt = DT, store_steps = 100)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.animate(L_norm = 1, Z_norm = 1)
#visualization.animate(xlim=[-xmax/2 * Å,xmax/2 * Å], animation_duration = 10, save_animation = False, fps = 30)


#for visualizing a single frame, use plot method instead of animate:
#visualization.plot(t = 0 ,xlim=[-xmax* Å,xmax* Å])
visualization.final_plot()
#visualization.plot(t = 160 * milliseconds,xlim=[-xmax* Å,xmax* Å])
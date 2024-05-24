import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, femtoseconds,ms,nanoseconds, microseconds,seconds,hbar, Hz,m,nm,m_e, Å

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
mass=7.016004 # Lithium
N= 3e5

acr = -0.33*nm

r_t = 3e-6*m   # 3 micro meters
print('r_t =', r_t)
omega_rho = 1.0e3*Hz # 1kHz
print('omega_rho =', omega_rho)
omega_z = 0
print('omega_z =', omega_z)

V0 = (hbar*omega_rho)/2
L = 4*r_t

#dimension-less variables
V0_tilde = V0/(2*hbar*omega_rho)
L_tilde = L/r_t


xmax = 100*r_t # 0.3 mm             # x-window size


def tho(t):
    return omega_rho*t

def eta(x):
    return x/r_t

def pot(particle):
    return V0*(1 - np.exp(-(particle.x/L)**2))
    return np.zeros_like(particle.x)

def initial_wavefunction(particle): # Initial wavefunction ground-state: a Gaussian 
    return np.sqrt(N)/(np.pi**(1/4) * np.sqrt(L_tilde)) * np.exp( -1*eta(particle.x)**2/(2*(L_tilde**2) ) )
    

def yeeto(psi,t,particle):
    
    a_z = np.zeros(particle.x.shape, dtype = np.complex128)
    for i in range(a_z.shape[0]):
        if particle.x[i] < 2*L:
            a_z[i] = 0*np.abs(psi[i])**2
        else:
            a_z[i] = 1.2*acr*np.abs(psi[i])**2
    return a_z


def yeet(psi,t,particle):
    return 0.00001*np.abs(psi)**2
    if t < 1 * microseconds:
        return 0.2*nm*abs(psi)**2
    elif t < 8 * microseconds:
        return 1.5*nm*abs(psi)**2
    else:
        return -0.2*nm*abs(psi)**2

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = m_e), 
                potential = pot, 
                spatial_ndim = 1, N = 500, extent = 2*xmax * 1)



#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#
total_time = 4000
#set the time dependent simulation
sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim.run(initial_wavefunction, total_time = total_time, dt = total_time/2000., store_steps = 200,non_linear_function=None)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.animate(xlim=[-xmax* 1,xmax* 1], animation_duration = 10, save_animation = True, fps = 30)


#for visualizing a single frame, use plot method instead of animate:
visualization.plot(t = 0 ,xlim=[-xmax* 1,xmax* 1])
visualization.plot(t = 160 * seconds,xlim=[-xmax* 1,xmax* 1])
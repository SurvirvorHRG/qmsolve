import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, femtoseconds, m_e, Å, m, seconds,kg,hbar

#=========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#
# Constants and parameters


mass = 1.44e-25 * kg # Mass of a 87Rb atom (kg)

omega = 2 * np.pi * 100 / seconds  # Trap frequency (rad/s)

g_acc = 9.81 * (m/ seconds / seconds)  # Acceleration due to gravity (m/s^2)

N = 1000  # Number of points in the spatial grid

L = 2*1e-5 * 1e10 # Spatial domain length (m)




dt = 1e-5  * seconds # Time step (s)

T = 1e-3   * seconds# Total simulation time (s)


a_s = 5.2e-9  * m # s-wave scattering length for Rb-87 (m)

interaction_strength = 4 * np.pi * hbar**2 * a_s / m  # Interaction strength (J·m)

 

# Initial wave function (ground state of the harmonic oscillator)

def psi_0(particle):

    return (mass * omega / (np.pi * hbar))**(1/4) * np.exp(-mass * omega * (particle.x)**2 / (2 * hbar))
#interaction potential


def gr(particle):

    return mass* g_acc * particle.x


def free(particle):
    return np.zeros_like(particle.x)

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = mass), 
                potential = gr, 
                spatial_ndim = 1, N = 500, extent = L * Å)


#=========================================================================================================#
# Define the wavefunction at t = 0  (initial condition)
#=========================================================================================================#


#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#

total_time = T
#set the time dependent simulation
sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim.run(psi_0, total_time = total_time, dt = dt, store_steps = 100)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.animate(xlim=[-L/2* Å,L/2* Å], animation_duration = 10, save_animation = True, fps = 30)


#for visualizing a single frame, use plot method instead of animate:
#visualization.plot(t = 5/4 * 0.9 * femtoseconds,xlim=[-15* Ã
,15* Ã
])
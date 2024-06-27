import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, femtoseconds, m_e, Å

#=========================================================================================================#
#First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#

#interaction potential
def harmonic_oscillator(particle):
    m = m_e

    return 0.5 * particle.x**2    +    0.5 * particle.y**2

def interaction(psi,t,particle):
    return - abs(psi)**2

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = m_e), 
                potential = harmonic_oscillator, 
                spatial_ndim = 2, N = 1000, extent = 200 *Å)



#=========================================================================================================#
# Define the wavefunction at t = 0  (initial condition)
#=========================================================================================================#

def initial_wavefunction(particle):
    
    return 0.5 * 1/np.cosh(np.sqrt(2)/4 * (particle.x + particle.y ))
    


#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 20
sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
sim.run(initial_wavefunction, total_time = total_time, dt = total_time/1000., store_steps = 400,non_linear_function=interaction)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
for i in range(11):
    visualization.plot3D(t = i * total_time/10, unit = 1)

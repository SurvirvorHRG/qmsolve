import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, femtoseconds, m_e, �

#=========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#

# File: ./examples1D/Solitons_in_phase_1D.py
# Run as    python3 bpm.py Solitons_in_phase_1D 1D
# Encounter of two solitons in phase, which traverse each other unaffected.
# During the collision, an interference pattern is generated.


import numpy as np

Nx = 500						# Grid points
Ny = Nx
dt = 0.001					# Evolution step
tmax = 5		            # End of propagation
xmax = 10 					# x-window size
ymax = xmax					# y-window size
images = 100				# number of .png images


def psi_0(particle):				# Initial wavefunction

# Two solitons heading each other in phase coincidence

	x01=-5
	x02=-x01
	v1=2
	v2=-v1
	fase1=0
	fase2=0
	f1 = (1/np.cosh(particle.x-x01))*np.exp(v1*1j*particle.x+1j*fase1)	# Soliton 1
	f2 = (1/np.cosh(particle.x-x02))*np.exp(v2*1j*particle.x+1j*fase2)	# Soliton 2
	f = f1+f2

	return f;
#interaction potential
def harmonic_oscillator(particle):
    m = m_e
    T = 0.6*femtoseconds
    w = 2*np.pi/T
    k = m* w**2
    return 0.5*k*particle.x**2 

def non_linear_f(psi):
    m = m_e
    T = 0.6*femtoseconds
    w = 2*np.pi/T
    k = m* w**2
    return -(abs(psi))**2

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = m_e), 
                potential = harmonic_oscillator, 
                spatial_ndim = 1, N = 500, extent = 20 * �)



#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#

total_time = 0.5 * femtoseconds
#set the time dependent simulation
sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim.run(psi_0,total_time = total_time,dt = total_time/1600.,store_steps = 800,non_linear_function = non_linear_f)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.animate(xlim=[-10* �,10* �], animation_duration = 10, save_animation = True, fps = 30)


#for visualizing a single frame, use plot method instead of animate:
#visualization.plot(t = 0.028 * femtoseconds,xlim=[-10* �,10* �])
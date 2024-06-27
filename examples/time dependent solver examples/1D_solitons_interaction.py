from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, nanoseconds,microseconds,nm,s,seconds, m,m_e, Å, J, Hz, kg, hbar, femtoseconds,picoseconds
import math




# File: ./examples1D/Soliton_Emission_B_1D.py
# Run as    python3 bpm.py Soliton_Emission_B_1D 1D
# Initially, a Gaussian wave packet is confined within a shallow trap. At a given
# time, a repulsive interaction is turned on and the wave starts expanding. Then,
# the sign of the nonlinear term is changed and the interaction becomes attractive.
# This results in the formation of a bunch of solitons that escape from the trap.


def psi_0(particle):                # Initial wavefunction: a Gaussian

    
    f= np.exp(-((particle.x)**2)/np.sqrt(2)/4)
    
    return f;
#interaction potential

def potential(particle):
    #V = -np.exp(-((particle.x)/4)**2)
    V = 0.2e-3*particle.x**6
    return V
def V(particle):
    #♥return np.zeros_like(particle.x)
    return +9.81*m_e*particle.x
    #return -np.exp(-((particle.x)/4)**2)

def V_non_linear(psi,t,particle):        
    
    #print(t)

    # The linear part of the potential is a shallow trap modeled by an inverted Gaussian
    # The nonlinear part is a cubic term whose sign and strength change abruptly in time.
    
    a0=0;  # initial (vanishing) nonlinear coefficient    
    a1=25;   # repulsive nonlinear coefficient for 3<t<8
    a2=-35;   # attractive nonlinear coefficient for t>8

    if t< 1 :
        V= a0*abs(psi)**2
    elif t<3 :
        V= a1*abs(psi)**2
    else:
        V= a2 *abs(psi)**2

    return V;



def free_fall_potential(particle):
    V = np.zeros_like(particle.x)
    return V




N_point = 1200

#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(m = m_e),
                potential=potential,
                spatial_ndim=1, N=N_point,extent=30 * Å)



def initial_wavefunction(particle):
    
    f= np.exp(-((particle.x)**2)/np.sqrt(2)/4)
    return f

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 10 
sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim.run(initial_wavefunction, total_time = total_time, dt = 0.0002, store_steps = 400, non_linear_function=V_non_linear)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.final_plot()
#visualization.plot(t = 0 * femtoseconds,xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2)
#visualization.plot(t = 0 ,unit = femtoseconds,contrast_vals=[0.5,1])
visualization.animate( animation_duration = 10, save_animation = True, fps = 30)
"""
visualization.animate(unit = femtoseconds,contrast_vals=[0.1,1])
for i in range(21):
    visualization.plot2D(t = i * total_time/20 ,unit = femtoseconds)
#visualization.plot_type = 'contour'
#visualization.animate(xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = False)
"""
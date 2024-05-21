from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, nanoseconds,microseconds,nm,s,seconds, m,m_e, Å, J, Hz, kg, hbar, femtoseconds,picoseconds
import math
import cupy as cp

# conversion units

hbar=1.054571596e-34
clight=299792458.0 # speed of light in m.s-1
echarge=1.602176462e-19 # electron charge in C
emass=9.10938188e-31 # electron mass in kg
pmass=1.67262158e-27 # proton mass in kg
uaumass=1.66053873e-27 # unified atomic unit of mass (1/12 of (12)C) in kg
epsilon0=1.0e7/(4*np.pi*clight*clight) # vacuum permittivity in C^2.kg-1.m-3.s^2
kBoltzmann=1.3806503e-23 # constante de boltzmann
aulength=4*np.pi*epsilon0*(hbar**2)/(emass*echarge*echarge)
auenergy=hbar*hbar/(emass*aulength*aulength) # Hartree energy in J
autime=hbar/auenergy
conv_au_fs=autime/1.0e-15
conv_C12_au=uaumass/emass
conv_au_fs=autime/1.0e-15
conv_au_ang=aulength/1.0e-10
conv_K_au=kBoltzmann/auenergy

# Parameters
mass=86.909  # Atoms mass Cs 132.905 , Rb 86.909 (united atomic unit of mass)
N=1000    # Number of condensed Bosons
a=5.2383     # s-wave scattering length - Rb 5.2383 , Cs 3.45 - (nm)

# Potentiel
l=6             # Radial index
#w0=30e-6   # Laser waist (mm) !1.0378725 pour l=1 0.0300185 pour l=6
#w1=30e-6    # Laser waist (mm) !1.0378725 pour l=1 0.0300185 pour l=6
#w0=30e-6   # Laser waist (mm) ! 30 microns pour l=1 
#w1=30e-6    # Laser waist (mm) !30 microns pour l=1 
w0=0.1e-6   #  18 microns pour l=6
w1=0.1e-6    #  18 microns pour l=6
#muc=173.3014     # Pot. chim. du condensat (nK) !173.3014 pour l=1 86.7018 pour l=6
muc=86.7018     # Pot. chim. du condensat (nK) !173.3014 pour l=1 86.7018 pour l=6
Power=1.0       # Laser Power (W)
delta=10.0      # Detuning / 2Pi (GHz)
Is=16.0         # Saturation intensity (W/m^2)
Gamma=6.0       # Natural width / 2Pi (MHz)

mass=mass*conv_C12_au
a=1*a/conv_au_ang
Power=Power*autime/auenergy
#w0=w0*1.0e-3/aulength
#w1=w1*1.0e-3/aulength

w0=w0*1.0/aulength
w1=w1*1.0/aulength
delta=2*np.pi*delta*conv_au_fs/1.0e6
Is=Is*autime*aulength*aulength/auenergy
Gamma=2*np.pi*Gamma*conv_au_fs/1.0e9
muc=muc*(conv_K_au/1.0e9)

factl=np.math.factorial(l)


# calcul of gint
gint = N*4*np.pi*a/mass
print(' gint = ',gint )
coeff1 = 2*(10**(-12))
coeff2 = 1*(10**(-12))
Ul=(Power*(Gamma**2)*(2**l))/(factl*4*np.pi*delta*Is*(w1**(2*l+2)))
U0=(Power*(Gamma**2)*(2**l))/(factl*4*np.pi*delta*Is*(w0**(2*l+2)))
print('U1 = ', Ul)
print('U0 = ', U0)
omega_l=(1/mass)*(((2**l)*mass*Ul))**(1/(l+1))

length_l=np.sqrt(1/(mass*omega_l))


#Characteristic sizes (au)

#omega_l=(one/mass)*(((two**l)*mass*Ul)**(one/real(l+1,real_8)))
Sizex=2*((muc/Ul)**(1/(2*l)))
Sizez=2*((muc/U0)**(1/(2*l)))
Sizey=Sizex
Rx=Sizex/2
Ry=Sizey/2
Rz=Sizez/2
omega_z = (2/Sizez)*np.sqrt(2*muc/mass)
length_z=np.sqrt(1/(mass*omega_z))
print(' Sizex = ',Sizex*(conv_au_ang/1.0e4),' microns.' )
print(' Sizey = ',Sizey*(conv_au_ang/1.0e4),' microns.')
print(' Sizez = ',Sizez*(conv_au_ang/1.0e4),' microns.')



# File: ./examples1D/Soliton_Emission_B_1D.py
# Run as    python3 bpm.py Soliton_Emission_B_1D 1D
# Initially, a Gaussian wave packet is confined within a shallow trap. At a given
# time, a repulsive interaction is turned on and the wave starts expanding. Then,
# the sign of the nonlinear term is changed and the interaction becomes attractive.
# This results in the formation of a bunch of solitons that escape from the trap.


def psi_0(particle):				# Initial wavefunction: a Gaussian

	
	f= np.exp(-((particle.x)**2)/np.sqrt(2)/4)
	
	return f;
#interaction potential
def V(particle):
    return -np.exp(-((particle.x)/4)**2)

def V_non_linear(psi,t,particle):		

	# The linear part of the potential is a shallow trap modeled by an inverted Gaussian
	# The nonlinear part is a cubic term whose sign and strength change abruptly in time.
	
	a0=0;  # initial (vanishing) nonlinear coefficient    
	a1=25;   # repulsive nonlinear coefficient for 3<t<8
	a2=-35;   # attractive nonlinear coefficient for t>8

	if t< 1 :
		V=a0*abs(psi)**2
	elif t<3 :
		V= a1*abs(psi)**2
	else:
		V= a2 *abs(psi)**2

	return V;



def free_fall_potential(particle):
    V = np.zeros_like(particle.x)
    return V

g = 9.81
def gravity_potential(particle):
    V = -g*N*mass*particle.z
    return V

def LG_potential(particle):
    #rho = np.sqrt(particle.x**2+particle.y**2)
    rho = np.sqrt(particle.x**2)

    V = (0.5**l)*omega_l*((rho/length_l)**(2*l))*np.exp(-2*(rho**2)/w0**2)
   # V = (0.5**l)*omega_l*((rho/length_l)**(2*l)) + (0.5**l)*omega_l*((particle.z/length_l)**(2*l))
    return V


def non_linear_potential(psi,t,particle):
   
    if t < (20  * femtoseconds):
        a = 5.2383
    elif t < (100 * femtoseconds):
        a = 10
    else:
        a = -5
        
    a= a/conv_au_ang
    gint = N*4*np.pi*a/mass
    return gint*((np.abs(psi))**2)

N_point = 1200

#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(),
                potential=V,
                spatial_ndim=1, N=N_point,extent=120* Å)



def initial_wavefunction(particle):
    
    f= np.exp(-((particle.x)**2)/np.sqrt(2)/4)
    return f
    #V = LG_potential(particle)
    #rho = np.sqrt(particle.x**2+particle.y**2)
    #psi = np.exp(-0.5*mass*(omega_l*rho**2 + omega_z*particle.z**2))
    """

    psi = np.zeros(particle.x.shape, dtype = np.complex128)
    for i in range(N_point):
        if muc > V[i]:
            psi[i] = np.sqrt( (muc - V[i]) / gint)
        else:
            psi[i] = 0
    return psi
"""



"""
eigenstates = H.solve( max_states = 32, method ='lobpcg-cupy')
print(eigenstates.energies)


visualization = visualization.init_visualization(eigenstates)
visualization.plot_type = 'contour'

#visualization.plot_eigenstate(0)

#visualization.plot_eigenstate(26)
visualization.animate()
"""


#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 10 
sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim.run(psi_0, total_time = total_time, dt = 0.0002, store_steps = 400, non_linear_function=V_non_linear)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
#visualization.plot(t = 0 * femtoseconds,xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2)
#visualization.plot(t = 0 ,unit = femtoseconds,contrast_vals=[0.5,1])
visualization.animate(xlim=[-60* Å,60* Å], animation_duration = 10, save_animation = True, fps = 30)
"""
visualization.animate(unit = femtoseconds,contrast_vals=[0.1,1])
for i in range(21):
    visualization.plot2D(t = i * total_time/20 ,unit = femtoseconds)
#visualization.plot_type = 'contour'
#visualization.animate(xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = False)
"""
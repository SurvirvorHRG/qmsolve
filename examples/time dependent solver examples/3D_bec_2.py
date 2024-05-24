from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, nanoseconds,microseconds,nm,s,seconds, m,m_e, Å, J, Hz, kg, hbar, femtoseconds,picoseconds
import math


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
l=1           # Radial index
w0=30e-6   # Laser waist (mm) !1.0378725 pour l=1 0.0300185 pour l=6
w1=30e-6    # Laser waist (mm) !1.0378725 pour l=1 0.0300185 pour l=6
#w0=30e-6   # Laser waist (mm) ! 30 microns pour l=1 
#w1=30e-6    # Laser waist (mm) !30 microns pour l=1 

#w0=10e-6   # Laser waist (mm) ! 30 microns pour l=2
#w1=10e-6    # Laser waist (mm) !30 microns pour l=2 
#w0=0.1e-6   #  18 microns pour l=6
#w1=0.1e-6    #  18 microns pour l=6
muc=173.3014     # Pot. chim. du condensat (nK) !173.3014 pour l=1 86.7018 pour l=6
#muc=86.7018     # Pot. chim. du condensat (nK) !173.3014 pour l=1 86.7018 pour l=6
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


#interaction potential
def free_fall_potential(particle):
    V = np.zeros_like(particle.x)
    return V

g = 9.81
def gravity_potential(particle):
    V = np.zeros_like(particle.x)
    V = -g*N*mass*particle.z
    V = g*N*mass*particle.z
    return V

def LG_potential(particle):
    rho = np.sqrt(particle.x**2+particle.y**2)

    V = (0.5**l)*omega_l*((rho/length_l)**(2*l))*np.exp(-2*(rho**2)/w0**2) + (0.5**l)*omega_l*((particle.z/length_l)**(2*l))*np.exp(-2*(particle.z**2)/w0**2)
   # V = (0.5**l)*omega_l*((rho/length_l)**(2*l)) + (0.5**l)*omega_l*((particle.z/length_l)**(2*l))
    return V


def non_linear_f(psi,t,particle):
    return gint*((np.abs(psi))**2)

N_point = 100

#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(),
                potential=gravity_potential,
                spatial_ndim=3, N=100,extent=500* Å,z_extent=500* Å)



def initial_wavefunction(particle):
    V = LG_potential(particle)
    #rho = np.sqrt(particle.x**2+particle.y**2)
    #psi = np.exp(-0.5*mass*(omega_l*rho**2 + omega_z*particle.z**2))

    psi = np.zeros(particle.x.shape, dtype = np.complex128)
    for i in range(N_point):
        for j in range(N_point):
            for k in range(N_point):
                if muc > V[i,j,k]:
                    psi[i,j,k] = np.sqrt( (muc - V[i,j,k]) / gint)
                else:
                    psi[i,j,k] = 0
    return psi

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


total_time = 500 * femtoseconds
sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
sim.run(initial_wavefunction, total_time = total_time, dt = (0.01 * femtoseconds), store_steps = 100,non_linear_function=None)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
#visualization.plot(t = 0 * femtoseconds,xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2)
#visualization.plot3D(t = 200* femtoseconds ,unit = femtoseconds,contrast_vals=[0.1,1])
#visualization.animate3D(unit = femtoseconds,contrast_vals=[0.1,1])
visualization.final_plot(L_norm = 1, Z_norm = 1,unit = femtoseconds,time = 'ns')
visualization.final_plot3D(L_norm = 1, Z_norm = 1,unit = femtoseconds,time = 'ns')
#for i in range(21):
    #visualization.plot3D(t = i * total_time/20 ,unit = femtoseconds)
#visualization.plot_type = 'contour'
#visualization.animate(xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = False)
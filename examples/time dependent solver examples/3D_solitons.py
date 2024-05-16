import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, femtoseconds, m_e, Å

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
l=1             # Radial index
w0=1.0378725    # Laser waist (mm) !1.0378725 pour l=1 0.0300185 pour l=6
w1=1.0378725    # Laser waist (mm) !1.0378725 pour l=1 0.0300185 pour l=6
muc=173.3014     # Pot. chim. du condensat (nK) !173.3014 pour l=1 86.7018 pour l=6
Power=1.0       # Laser Power (W)
delta=10.0      # Detuning / 2Pi (GHz)
Is=16.0         # Saturation intensity (W/m^2)
Gamma=6.0       # Natural width / 2Pi (MHz)

mass=mass*conv_C12_au
a=1*a/conv_au_ang
Power=Power*autime/auenergy
w0=w0*1.0e-3/aulength
w1=w1*1.0e-3/aulength
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

#=========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#=========================================================================================================#

#interaction potential
def harmonic_oscillator(particle):
    m = m_e
    T = 0.6*femtoseconds
    w = 2*np.pi/T
    k = m* w**2
    return 0.5 * k * particle.x**2 

N = 500
#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = m_e), 
                potential = harmonic_oscillator, 
                spatial_ndim = 1, N = 500, extent = 4 * Å)


#=========================================================================================================#
# Define the wavefunction at t = 0  (initial condition)
#=========================================================================================================#

def initial_wavefunction(particle):
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x
    V = harmonic_oscillator(particle)
    psi = np.zeros(particle.x.shape, dtype = np.complex128)
    for i in range(N):
        if muc > V[i]:
            psi[i] = np.sqrt( (muc - V[i]) / gint)
        else:
            psi[i] = 0
    return psi

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#

total_time = 1.8 * femtoseconds
#set the time dependent simulation
sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim.run(initial_wavefunction, total_time = total_time, dt = total_time/1600., store_steps = 800)

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.animate(xlim=[-2* Å,2* Å], animation_duration = 10, save_animation = True, fps = 30)


#for visualizing a single frame, use plot method instead of animate:
#visualization.plot(t = 5/4 * 0.9 * femtoseconds,xlim=[-15* Å,15* Å])
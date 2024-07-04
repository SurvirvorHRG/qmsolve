from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, milliseconds,microseconds,nm,s,seconds, meters,m_e, Å, J, Hz, kg
import math
import scipy.special as sc

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
N=1e6    # Number of condensed Bosons
a=5.2383     # s-wave scattering length - Rb 5.2383 , Cs 3.45 - (nm)

# Potentiel


#l=1            # Radial index
#w0=0.7936514     # Laser waist (mm) !0.7936514 pour l=1 0.7865231 pour l=6
#w1=1.7746586     # Laser waist (mm) !1.7746586 pour l=1 0.0738672 pour l=6
#muc=173.3320     # Pot. chim. du condensat (nK) !173.3320 pour l=1 144.6547 pour l=6


l=6            # Radial index
w0=0.7865231     # Laser waist (mm) !0.7936514 pour l=1 0.7865231 pour l=6
w1=0.0738672     # Laser waist (mm) !1.7746586 pour l=1 0.0738672 pour l=6
muc=144.6547     # Pot. chim. du condensat (nK) !173.3320 pour l=1 144.6547 pour l=6


Power=1.0       # Laser Power (W)
delta=10.0      # Detuning / 2Pi (GHz)
Is=16.0         # Saturation intensity (W/m^2)
Gamma=6.0       # Natural width / 2Pi (MHz)
xmin = -2.0
xmax = 2.0 
ymin = xmin
ymax = xmax
zmin = -40.0
zmax = 40.0

mass=mass*conv_C12_au
a=10*a/conv_au_ang
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
U0=(Power*(Gamma**2)*(2**l))/(factl*4*np.pi*delta*Is*(w0**(2*l+2)))
Ul=(Power*(Gamma**2)*(2**l))/(factl*4*np.pi*delta*Is*(w1**(2*l+2)))
print('U0 = ', U0)
print('U1 = ', Ul)




#Characteristic sizes (au)

#omega_l=(one/mass)*(((two**l)*mass*Ul)**(one/real(l+1,real_8)))
omega_l=(1/mass)*(((2**l)*mass*Ul)**(1/(l+1)))
Sizex=2*((muc/U0)**(1/(2*l)))
Sizey=Sizex
Sizez=2*((muc/Ul)**(1/(2*l)))

Rx=Sizex/2
Ry=Sizey/2
Rz=Sizez/2

print(' Sizex = ',Sizex*(conv_au_ang/1.0e4),' microns.' )
print(' Sizey = ',Sizey*(conv_au_ang/1.0e4),' microns.')
print(' Sizez = ',Sizez*(conv_au_ang/1.0e4),' microns.')

#facteur_gl=exp(gammln(one/real(l,real_8))+gammln(one/(two*real(l,real_8)))- gammln(one+(three/(two*real(l,real_8)))))/(four*real(l*l,real_8))
facteur_gl=np.exp(sc.gammaln(1/(l))+sc.gammaln(1/(2*(l))) - sc.gammaln(1+(3/(2*(l)))))/(4*(l*l))
R0=Rz*(facteur_gl**(1/3))

print('Gamma_l = ',facteur_gl)

print('R0 = ',R0*(conv_au_ang/1.0e4),' microns.')

print('Rx/R0 = ',Rx/R0)
print('Ry/R0 = ',Ry/R0)
print('Rz/R0 = ',Rz/R0)

print('Volume = ',4*np.pi*facteur_gl*Rx*Ry*Rz*((aulength*1.0e6)**3),' microns**3.')


omega_z = (2/Sizez)*np.sqrt(2*muc/mass)
length_l=np.sqrt(1/(mass*omega_l))
length_z=np.sqrt(1/(mass*omega_z))

print(' omega_l = ',omega_l,' au')
print(' omega_z = ',omega_z,' au')

print('omega_l / 2Pi = ',omega_l/(2*np.pi*1.0e-15*conv_au_fs),' Hz.')
print('omega_z / 2Pi = ',omega_z/(2*np.pi*1.0e-15*conv_au_fs),' Hz.')
print('length_l = ',length_l*aulength*1.0e6,' micro-m.') 
print('length_z = ',length_z*aulength*1.0e6,' micro-m.') 


#Define grid boundaries (au)
xmin=xmin*R0
xmax=xmax*R0
ymin=ymin*R0
ymax=ymax*R0
zmin=zmin*R0
zmax=zmax*R0
#interaction potential

def potential(particle):
    V = np.zeros_like(particle.x)
    return V
g = 9.8065 * (meters / seconds / seconds)

def gravity(particle):
    return N * g * mass * particle.z
    
def LG_potential(particle):
    #rho = np.sqrt(particle.x**2+particle.y**2)
    x = particle.x 
    y = 0 
    z = 0                                                                              
    Vtrap = (0.5**l)*omega_l*((np.sqrt((x)**2+(y)**2)/length_l)**((2*l)))*np.exp(-2*((x)**2+(y)**2)/(w0**2))
    + (0.5**l)*omega_l*((z/length_l)**((2*l)))*np.exp(-2*((z)**2)/(w0**2))
    return Vtrap


def non_linear_f(psi,t,particle):
    return gint*abs(psi)**2


def non_linear_f2(psi,t,particle):
    if t < 3 * milliseconds:
        return gint*((np.abs(psi))**2)
    else:
        return -gint*((np.abs(psi))**2)


N = 2000
extent =4 * xmax
#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=LG_potential,
                spatial_ndim=1, N=N,extent=extent)



def initial_wavefunction(particle):
    V = LG_potential(particle)
    psi = np.zeros(particle.x.shape, dtype = np.complex128)
    for i in range(N):
        if muc > V[i]:
            psi[i] = np.sqrt( (muc - V[i]) / gint)
        else:
            psi[i] = 0
    return psi


def initial_wavefunction_1(particle):
    #psi = initial_wavefunction(particle)
    
    V = LG_potential(particle)
    return V

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 0.01 * seconds
dt = 1e-7 * seconds
stored = 100
sim = TimeSimulation(hamiltonian = H, method = "nonlinear-split-step")
sim.method.split_step.set_nonlinear_term(non_linear_f)
sim.run(initial_wavefunction, total_time = total_time, dt = dt, store_steps = stored)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.final_plot()
#visualization.animate(save_animation=True)
for i in range(11):
    visualization.plot1D(t = i * total_time/10)

from tvtk.util import ctf
import numpy as np
from qmsolve import visualization
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, milliseconds,microseconds,nm,s,seconds, meters,m_e, Å, J, Hz, kg, hbar, femtoseconds,picoseconds
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
l=6            # Radial index
#w0=0.7936514     # Laser waist (mm) !0.7936514 pour l=1 0.7865231 pour l=6
#w1=1.7746586     # Laser waist (mm) !1.7746586 pour l=1 0.0738672 pour l=6
#muc=173.3320     # Pot. chim. du condensat (nK) !173.3320 pour l=1 144.6547 pour l=6

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
    return g* mass * particle.z
    
def LG_potential(particle):
    #rho = np.sqrt(particle.x**2+particle.y**2)
    x = particle.x 
    y = particle.y 
    z = particle.z 
    
   # V = (0.5**l)*omega_l*((rho/length_l)**(2*l))*np.exp(-2*(rho**2)/w0**2) + (0.5**l)*omega_l*((particle.z/length_l)**(2*l))*np.exp(-2*(particle.z**2)/w0**2) 
                                                                                
    Vtrap = U0*((np.sqrt((x)**2+(y)**2))**((2*l)))*np.exp(-2*((x)**2+(y)**2)/(w0**2)) + Ul*((z)**((2*l)))*np.exp(-2*((z)**2)/(w1**2))

    return Vtrap


def non_linear_f(psi,t,particle):
    return -gint*((np.abs(psi))**2)


def non_linear_f2(psi,t,particle):
    if t < 1 * milliseconds:
        return 0*((np.abs(psi))**2)
    elif t < 3 * milliseconds:
        return gint*((np.abs(psi))**2)
    else:
        return -gint*((np.abs(psi))**2)


N = 128
Nz = 512
extent = 2 * xmax
z_extent = 2* zmax
#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(m = mass),
                potential=potential,
                spatial_ndim=3, N=N,Nz = Nz,extent=extent,z_extent=z_extent)



def initial_wavefunction(particle):
    V = LG_potential(particle)
    #rho = np.sqrt(particle.x**2+particle.y**2)
    #psi = np.exp(-0.5*mass*(omega_l*rho**2 + omega_z*particle.z**2))
    """
    psi = np.zeros(particle.x.shape, dtype = np.complex128)
    for i in range(N):
        for j in range(N):
            if muc > V[i,j]:
                psi[i,j] = np.sqrt( (muc - V[i,j]) / gint)
            else:
                psi[i,j] = 0
    return psi
    """
    psi = np.zeros(particle.x.shape, dtype = np.complex128)
    for i in range(N):
        for j in range(N):
            for k in range(Nz):
                if muc > V[i,j,k]:
                    psi[i,j,k] = np.sqrt( (muc - V[i,j,k]) / gint)
                else:
                    psi[i,j,k] = 0
    return psi

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 0.01 * seconds
dt = 1e-6 * seconds
#total_time = dt
#dt = min(dt/omega_l,dt/omega_z)
sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
sim.run(initial_wavefunction, total_time = total_time, dt = dt, store_steps = 20,non_linear_function=non_linear_f)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
visualization.plot3D(0, unit = milliseconds)
visualization.plot(0, unit = milliseconds)
#visualization.plot(t = 0 * femtoseconds,xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2)
#visualization.plot2D(t = 0 ,unit = femtoseconds,contrast_vals=[0.9,1])
#for i in range(21):
    #visualization.plot2D_xy(t = i * total_time/20, unit = milliseconds)
#visualization.animate(unit = femtoseconds,contrast_vals=[0.99,1])
#visualization.plot_type = 'contour'
#visualization.animate(xlim=[-50* Å,50* Å], ylim=[-50* Å,50* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = True)
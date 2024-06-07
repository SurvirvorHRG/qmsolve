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
Ntot = 1e6
#n0eq = 0.1*Ntot
n0eq = Ntot
N=1000    # Number of condensed Bosons
a=5.2383     # s-wave scattering length - Rb 5.2383 , Cs 3.45 - (nm)

# Potentiel
l=6           # Radial index
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
Plg = 1.0
Pn = 1.0

mass=mass*conv_C12_au
a=10*a/conv_au_ang
Power=Power*autime/auenergy
Plg = Plg*autime/auenergy
Pn = Pn*autime/auenergy
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
#Ul=(Power*(Gamma**2)*(2**l))/(factl*4*np.pi*delta*Is*(w1**(2*l+2)))
#U0=(Power*(Gamma**2)*(2**l))/(factl*4*np.pi*delta*Is*(w0**(2*l+2)))
#print('U1 = ', Ul)
#print('U0 = ', U0)



#Characteristic sizes (au)

#omega_l=(one/mass)*(((two**l)*mass*Ul)**(one/real(l+1,real_8)))
gl = (1/((2*l)**2))*np.exp(math.lgamma(1/l)*math.lgamma(1/(2*l)) - math.lgamma((2*l + 3)/(2*l)))
print('gl = ',gl)
R0 = 15/(conv_au_ang/1.0e4)
Rl = R0/(gl**(1/3))
print(' Rl = ',Rl*(conv_au_ang/1.0e4),' microns.' )
Dx = 2*Rl
Dz = Dx
Dy = Dx
print( 'Dx = ',Dx*conv_au_ang/1.0e4,' microns.')

print('Dy = ',Dy*conv_au_ang/1.0e4,' microns.')

print('Dz = ',Dz*conv_au_ang/1.0e4,' microns.')
Vl = 4*np.pi*gl*(Rl**3)
print('Vl = ',Vl)
#Sizex=2*((muc/Ul)**(1/(2*l)))
#Sizez=2*((muc/U0)**(1/(2*l)))
#Sizey=Sizex
#Rx=Sizex/2
#Ry=Sizey/2
#Rz=Sizez/2

#print(' Sizex = ',Sizex*(conv_au_ang/1.0e4),' microns.' )
#print(' Sizey = ',Sizey*(conv_au_ang/1.0e4),' microns.')
#print(' Sizez = ',Sizez*(conv_au_ang/1.0e4),' microns.')

#Calcul de U0 et U1
gi=(4*np.pi*a)/mass
print( 'gi = ',gi)
B=((2*l+3)*gi*n0eq/(4*np.pi))*np.exp(math.lgamma((2*l+3)/(2*l))-math.lgamma(1/l)-math.lgamma((2*l+1)/(2*l)))
print(' B = ',B)
U0=B*((2/Dx)**(2*l+2))*(2/Dz)
print('U0 = ',U0,' au')
U1=B*((2/Dx)**2)*((2/Dz)**(2*l+1))
print( 'U1 = ',U1,' au')

 

# Calcul de en0

gamal=np.sqrt(np.pi)*np.exp(math.lgamma((2*l+1)/(2*l))-math.lgamma((3*l+1)/(2*l)))

en0=((2/mass)**(l/(l+1)))*(U0**(1/(l+1)))*((np.pi/(2*gamal))**((2*l)/(l+1)))+ ((np.pi*(U1**(1/(2*l))))/(gamal*2*np.sqrt(2*mass)))**((2*l)/(l+1))

print(' énergie du fondamental* = ',en0,' au')

alpha=((((2*l+3)*gi)/(4*np.pi))*np.exp(math.lgamma((2*l+3)/(2*l))-math.lgamma(1/l)-math.lgamma((2*l+1)/(2*l)))*(U0**(1/l))*(U1**(1/(2*l))))**(((2*l)/(2*l+3)))
nu=(en0/alpha)**((2*l+3)/(2*l))
# Calcul de mu

muceq=alpha*(n0eq+nu)**((2*l)/(2*l+3))

print(' potentiel chimique du condensat à l équilibre = ',muceq,' au')

 

#write(*,*) abs(one-U1/(muceq/((Dz/two)**(two*l))))+abs(one-U0/(muceq/((Dx/two)**(two*l))))

#write(*,*)

#do while ((abs(one-U1/(muceq/((Dz/two)**(two*l))))+abs(one-U0/(muceq/((Dx/two)**(two*l))))) > 1.0E-15_real_8 )

 

U1=muceq/((Dz/2)**(2*l))

U0=muceq/((Dx/2)**(2*l))
print('U1 = ', U1)
print('U0 = ', U0)

 

#! Re-Calcul de en0

en0=((2/mass)**(l/(l+1)))*(U0**(1/(l+1)))*((np.pi/(2*gamal))**((2*l)/(l+1)))+ ((np.pi*(U1**(1/(2*l))))/(gamal*2*np.sqrt(2*mass)))**((2*l)/(l+1))

print(' énergie du fondamental = ',en0,' au')

print(' énergie du fondamental = ',en0/(conv_K_au/1.0e9),' nK')

 

alpha=((((2*l+3)*gi)/(4*np.pi))*np.exp(math.lgamma((2*l+3)/(2*l))-math.lgamma(1/l)-math.lgamma((2*l+1)/(2*l)))*(U0**(1/l))*(U1**(1/(2*l))))**(((2*l)/(2*l+3)))

 

nu=(en0/alpha)**((2*l+3)/(2*l))

 

#! calcul de muceq

muceq=alpha*(n0eq+nu)**((2*l)/(2*l+3))

print(' potentiel chimique du condensat à l équilibre = ',muceq,' au')

print(' potentiel chimique du condensat à l équilibre en nanok = ',muceq/(conv_K_au/1.0e9))

#print(abs(one-U1/(muceq/((Dz/two)**(two*l))))+abs(one-U0/(muceq/((Dx/two)**(two*l)))))

 
 

print('nu   = ',nu)

print('n0eq = ',n0eq)

 

#! calcul des waists

waist1=(((Gamma**2)*Plg*(2**l))/(np.exp(math.lgamma(l+1))*U1*4*np.pi*delta*Is))**(1/(2*l+2))

waist0=(((Gamma**2)*Pn*(2**l))/(np.exp(math.lgamma(l+1))*4*np.pi*delta*Is*U0))**(1/(2*l+2))

print('waist0 = ',waist0*aulength*1.0e6,' microns.')

print('waist1 = ',waist1*aulength*1.0e6,' microns.')

omega_l=(1/mass)*(((2**l)*mass*U0))**(1/(l+1))

length_l=np.sqrt(1/(mass*omega_l))

omega_z = (2/Dz)*np.sqrt(2*muc/mass)
length_z=np.sqrt(1/(mass*omega_z))
print(' omega_l = ',omega_l,' au')
print(' omega_z = ',omega_z,' au')


#interaction potential
def free_fall_potential(particle):
    V = np.zeros_like(particle.x)
    return V

g = 9.81
def gravity_potential(particle):
    #☻V = np.zeros_like(particle.x)
    V = -g*mass*particle.z
    #V = g*N*mass*particle.z
    return V

def LG_potential(particle):
    rho = np.sqrt(particle.x**2+particle.y**2)

    V = (0.5**l)*omega_l*((rho/length_l)**(2*l))*np.exp(-2*(rho**2)/waist0**2) + (0.5**l)*omega_l*((particle.z/length_l)**(2*l))*np.exp(-2*(particle.z**2)/waist1**2)
   # V = (0.5**l)*omega_l*((rho/length_l)**(2*l)) + (0.5**l)*omega_l*((particle.z/length_l)**(2*l))
    return V


def non_linear_f2(psi,t,particle):
    return  - gi*np.abs(psi)**2

def non_linear_f(psi,t,particle):
    
    
    l = 1
    
    if t < 0.2 * microseconds:
        l = 1
    elif t < 0.4 * microseconds:
        l = 3
    else:
        l = 6
    
    #print(l)
    gl = (1/((2*l)**2))*np.exp(math.lgamma(1/l)*math.lgamma(1/(2*l)) - math.lgamma((2*l + 3)/(2*l)))
    #print('gl = ',gl)
    R0 = 15/(conv_au_ang/1.0e4)
    Rl = R0/(gl**(1/3))
    #print(' Rl = ',Rl*(conv_au_ang/1.0e4),' microns.' )
    Dx = 2*Rl
    Dz = Dx
    Dy = Dx
    #print( 'Dx = ',Dx*conv_au_ang/1.0e4,' microns.')

    #print('Dy = ',Dy*conv_au_ang/1.0e4,' microns.')

    #print('Dz = ',Dz*conv_au_ang/1.0e4,' microns.')
    Vl = 4*np.pi*gl*(Rl**3)
    #print('Vl = ',Vl)

    #Calcul de U0 et U1
    gi=(4*np.pi*a)/mass
    #print( 'gi = ',gi)
    B=((2*l+3)*gi*n0eq/(4*np.pi))*np.exp(math.lgamma((2*l+3)/(2*l))-math.lgamma(1/l)-math.lgamma((2*l+1)/(2*l)))
    #print(' B = ',B)
    U0=B*((2/Dx)**(2*l+2))*(2/Dz)
    #print('U0 = ',U0,' au')
    U1=B*((2/Dx)**2)*((2/Dz)**(2*l+1))
    #print( 'U1 = ',U1,' au')

     

    # Calcul de en0

    gamal=np.sqrt(np.pi)*np.exp(math.lgamma((2*l+1)/(2*l))-math.lgamma((3*l+1)/(2*l)))

    en0=((2/mass)**(l/(l+1)))*(U0**(1/(l+1)))*((np.pi/(2*gamal))**((2*l)/(l+1)))+ ((np.pi*(U1**(1/(2*l))))/(gamal*2*np.sqrt(2*mass)))**((2*l)/(l+1))

    #print(' énergie du fondamental* = ',en0,' au')

    alpha=((((2*l+3)*gi)/(4*np.pi))*np.exp(math.lgamma((2*l+3)/(2*l))-math.lgamma(1/l)-math.lgamma((2*l+1)/(2*l)))*(U0**(1/l))*(U1**(1/(2*l))))**(((2*l)/(2*l+3)))
    nu=(en0/alpha)**((2*l+3)/(2*l))
    # Calcul de mu

    muceq=alpha*(n0eq+nu)**((2*l)/(2*l+3))

    #print(' potentiel chimique du condensat à l équilibre = ',muceq,' au')

     

     

    U1=muceq/((Dz/2)**(2*l))

    U0=muceq/((Dx/2)**(2*l))
    #print('U1 = ', U1)
    #print('U0 = ', U0)

     

    #! Re-Calcul de en0

    en0=((2/mass)**(l/(l+1)))*(U0**(1/(l+1)))*((np.pi/(2*gamal))**((2*l)/(l+1)))+ ((np.pi*(U1**(1/(2*l))))/(gamal*2*np.sqrt(2*mass)))**((2*l)/(l+1))

    #print(' énergie du fondamental = ',en0,' au')

    #print(' énergie du fondamental = ',en0/(conv_K_au/1.0e9),' nK')

     

    alpha=((((2*l+3)*gi)/(4*np.pi))*np.exp(math.lgamma((2*l+3)/(2*l))-math.lgamma(1/l)-math.lgamma((2*l+1)/(2*l)))*(U0**(1/l))*(U1**(1/(2*l))))**(((2*l)/(2*l+3)))

     

    nu=(en0/alpha)**((2*l+3)/(2*l))

     

    #! calcul de muceq

    muceq=alpha*(n0eq+nu)**((2*l)/(2*l+3))

    #print(' potentiel chimique du condensat à l équilibre = ',muceq,' au')

    #print(' potentiel chimique du condensat à l équilibre en nanok = ',muceq/(conv_K_au/1.0e9))

     
     

    #print('nu   = ',nu)

    #print('n0eq = ',n0eq)

     

    #! calcul des waists

    waist1=(((Gamma**2)*Plg*(2**l))/(np.exp(math.lgamma(l+1))*U1*4*np.pi*delta*Is))**(1/(2*l+2))

    waist0=(((Gamma**2)*Pn*(2**l))/(np.exp(math.lgamma(l+1))*4*np.pi*delta*Is*U0))**(1/(2*l+2))

    #print('waist0 = ',waist0*aulength*1.0e6,' microns.')

    #print('waist1 = ',waist1*aulength*1.0e6,' microns.')

    omega_l=(1/mass)*(((2**l)*mass*U0))**(1/(l+1))

    length_l=np.sqrt(1/(mass*omega_l))

    omega_z = (2/Dz)*np.sqrt(2*muc/mass)
    length_z=np.sqrt(1/(mass*omega_z))
    #print(' omega_l = ',omega_l,' au')
    #print(' omega_z = ',omega_z,' au')
    import cupy as cp 
    #rho = np.sqrt(particle.x**2+particle.y**2)
    #V = (0.5**l)*omega_l*((rho/length_l)**(2*l))*np.exp(-2*(rho**2)/waist0**2) + (0.5**l)*omega_l*((particle.z/length_l)**(2*l))*np.exp(-2*(particle.z**2)/waist1**2)
    rho = np.sqrt(particle.x**2+particle.y**2)
    V = (0.5**l)*omega_l*((rho/length_l)**(2*l))*np.exp(-2*(rho**2)/waist0**2) + (0.5**l)*omega_l*((particle.z/length_l)**(2*l))*np.exp(-2*(particle.z**2)/waist1**2)
    return cp.array(V) - gi*np.abs(psi)**2

N_point = 100

#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(),
                potential=free_fall_potential,
                spatial_ndim=3, N=N_point,extent=1000000* Å,z_extent=1000000* Å)



def initial_wavefunction(particle):
    V = LG_potential(particle)
    #rho = np.sqrt(particle.x**2+particle.y**2)
    #psi = np.exp(-0.5*mass*(omega_l*rho**2 + omega_z*particle.z**2))

    psi = np.zeros(particle.x.shape, dtype = np.complex128)
    for i in range(N_point):
        for j in range(N_point):
            for k in range(N_point):
                if muceq > V[i,j,k]:
                    psi[i,j,k] = np.sqrt( (muceq - V[i,j,k]) / gi)
                else:
                    psi[i,j,k] = 0
    return psi



#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 2 * microseconds
sim = TimeSimulation(hamiltonian = H, method = "split-step-cupy")
sim.run(initial_wavefunction, total_time = total_time, dt = (0.001 * microseconds), store_steps = 100,non_linear_function=non_linear_f2,m = mass,g = 9.81)


#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)
#visualization.animate(unit = microseconds,time = 'microseonds',contrast_vals=[0.1,0.25])
#visualization.plot3D(t = 0, unit = microseconds)
#for i in range(21):
    #visualization.plot3D(t = i * total_time/20, unit = microseconds)
#visualization.plot(t = 0 * femtoseconds,xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2)
#visualization.plot3D(t = 200* femtoseconds ,unit = femtoseconds,contrast_vals=[0.1,1])
#visualization.plot3D(t = 0,unit = nanoseconds)
#visualization.plot(t = 0,unit = nanoseconds)
visualization.final_plot3D(L_norm = 1, Z_norm = 1,unit = microseconds,time = 'microseconds')
#visualization.final_plot3D(L_norm = 1, Z_norm = 1,unit = femtoseconds,time = 'ns')
#for i in range(21):
    #visualization.plot3D(t = i * total_time/20 ,unit = femtoseconds)
#visualization.plot_type = 'contour'
#visualization.animate(xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, save_animation = False)

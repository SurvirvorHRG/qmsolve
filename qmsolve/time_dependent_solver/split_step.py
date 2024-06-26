import numpy as np
from .method import Method
import time
from ..util.constants import hbar, Å, femtoseconds
from ..particle_system import SingleParticle, TwoParticles
import progressbar

"""
Split-operator method for the Schrödinger equation.

Prototype and original implementation:
https://github.com/marl0ny/split-operator-simulations

References:
https://www.algorithm-archive.org/contents/
split-operator_method/split-operator_method.html
https://en.wikipedia.org/wiki/Split-step_method
"""

class SplitStep(Method):
    def __init__(self, simulation):

        self.simulation = simulation
        self.H = simulation.H
        self.simulation.Vmin = np.amin(self.H.Vgrid)
        self.simulation.Vmax = np.amax(self.H.Vgrid)
        self.H.particle_system.compute_momentum_space(self.H)
        self.p2 = self.H.particle_system.p2


    def run(self, initial_wavefunction, total_time, dt, store_steps = 1, non_linear_function = None ,norm = False,g = 1):

        self.simulation.store_steps = store_steps
        dt_store = total_time/store_steps
        self.simulation.total_time = total_time

        Nt = int(np.round(total_time / dt))
        Nt_per_store_step = int(np.round(dt_store / dt))
        self.simulation.Nt_per_store_step = Nt_per_store_step

        #time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
        self.simulation.dt = dt_store/Nt_per_store_step
        Ψ = 0
        if isinstance(self.simulation.H.particle_system ,SingleParticle):
            if self.H.ndim == 3:
                Ψ = np.zeros((store_steps + 1, self.H.N,self.H.N, self.H.Nz), dtype = np.complex128)
            else:
                Ψ = np.zeros((store_steps + 1, *([self.H.N] *self.H.ndim )), dtype = np.complex128)

        elif isinstance(self.simulation.H.particle_system,TwoParticles):
            Ψ = np.zeros((store_steps + 1, *([self.H.N] * 2)), dtype = np.complex128)

        Ψ[0] = np.array(initial_wavefunction(self.H.particle_system))



        m = self.H.particle_system.m


        Ur = np.exp(-0.5j*(self.simulation.dt/hbar)*np.array(self.H.Vgrid))
        Uk = np.exp(-0.5j*(self.simulation.dt/(m*hbar))*self.p2)
        
        


        t0 = time.time()
        bar = progressbar.ProgressBar()
        t_count = 0
        for i in bar(range(store_steps)):
            
            tmp = np.copy(Ψ[i])
            #Ur = np.exp(-0.5j*(self.simulation.dt/hbar)*(np.array(self.H.Vgrid) + non_linear_function(tmp)))
            #Ur *= np.exp(-0.5j*(self.simulation.dt/hbar)*non_linear_function(tmp))
            for j in range(Nt_per_store_step):
                t_count += 1
                t = t_count*self.simulation.dt
                if non_linear_function is not None:
                    Ur = np.exp(-0.5j*(self.simulation.dt/hbar)*(np.array(self.H.Vgrid) + non_linear_function(tmp,t,self.H.particle_system)))
                #Ur *= np.exp(-0.5j*(self.simulation.dt/hbar)*non_linear_function(tmp))
                #exp_g = np.exp(1j*m*g*t/hbar * (self.simulation.H.particle_system.x + (g*t**2)/6 ) )
                exp_g = 1
                c = np.fft.fftshift(np.fft.fftn(Ur*exp_g*tmp))
                tmp = Ur*np.fft.ifftn( np.fft.ifftshift(Uk*c))
                if norm:
                    tmp = tmp / np.sqrt(np.sum(tmp * np.conj(tmp)))
            Ψ[i+1] = tmp
        print("Took", time.time() - t0)



        self.simulation.Ψ = Ψ
        if norm:
            self.simulation.Ψmax = 1.
        else:
            self.simulation.Ψmax = np.amax(np.abs(Ψ))




class SplitStepCupy(Method):
    def __init__(self, simulation):

        self.simulation = simulation
        self.H = simulation.H
        self.simulation.Vmin = np.amin(self.H.Vgrid)
        self.simulation.Vmax = np.amax(self.H.Vgrid)

        self.H.particle_system.compute_momentum_space(self.H)
        self.p2 = self.H.particle_system.p2


    def run(self, initial_wavefunction, total_time, dt, store_steps = 1, non_linear_function = None,norm = False, g = 1):

        import cupy as cp 

        self.p2 = cp.array(self.p2)
        self.simulation.store_steps = store_steps
        dt_store = total_time/store_steps
        self.simulation.total_time = total_time

        Nt = int(np.round(total_time / dt))
        Nt_per_store_step = int(np.round(dt_store / dt))
        self.simulation.Nt_per_store_step = Nt_per_store_step

        #time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
        self.simulation.dt = dt_store/Nt_per_store_step
        Ψ = 0
        if self.H.ndim == 3:
            Ψ = cp.zeros((store_steps + 1, self.H.N,self.H.N, self.H.Nz), dtype = cp.complex128)
        else:
            Ψ = cp.zeros((store_steps + 1, *([self.H.N] *self.H.ndim )), dtype = cp.complex128)
            
        Ψ[0] = cp.array(initial_wavefunction(self.H.particle_system))



        m = self.H.particle_system.m


        Ur = cp.exp(-0.5j*(self.simulation.dt/hbar)*cp.array(self.H.Vgrid))
        Uk = cp.exp(-0.5j*(self.simulation.dt/(m*hbar))*self.p2)

        t0 = time.time()
        bar = progressbar.ProgressBar()
        t_count = 0
        for i in bar(range(store_steps)):
            tmp = cp.copy(Ψ[i])
            #Ur = cp.exp(-0.5j*(self.simulation.dt/hbar)*(cp.array(self.H.Vgrid) + non_linear_function(tmp)))
            #Ur *= cp.exp(-0.5j*(self.simulation.dt/hbar)*non_linear_function(tmp))
            for j in range(Nt_per_store_step):
                t = t_count*self.simulation.dt
                #exp_g = cp.exp(1j*m*g*t/hbar * cp.array((self.simulation.H.particle_system.z + (g*t**2)/6 )) )
                exp_g = 1
                tmp = exp_g * tmp
                if non_linear_function is not None:
                    Ur = cp.exp(-0.5j*(self.simulation.dt/hbar)*(cp.array(self.H.Vgrid) + cp.array(non_linear_function(tmp,t,self.H.particle_system))))
                #Ur *= cp.exp(-0.5j*(self.simulation.dt/hbar)*non_linear_function(tmp))

                c = cp.fft.fftshift(cp.fft.fftn(Ur*tmp))
                tmp = Ur*cp.fft.ifftn( cp.fft.ifftshift(Uk*c))
                if norm:
                    tmp = tmp / cp.sqrt(cp.sum(tmp*cp.conj(tmp)))
            Ψ[i+1] = tmp
        print("Took", time.time() - t0)



        self.simulation.Ψ = Ψ.get()
        if norm:
            self.simulation.Ψmax = 1.
        else:
            self.simulation.Ψmax = np.amax(np.abs(self.simulation.Ψ ))
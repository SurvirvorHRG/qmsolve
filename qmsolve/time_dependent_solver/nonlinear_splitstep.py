import numpy as np
from .method import Method
import time
from ..util.constants import hbar, Å, femtoseconds, m_e, seconds
from ..particle_system import SingleParticle, TwoParticles
import progressbar
from typing import Union, Callable, Tuple
import scipy.constants as const

# Constants and parameters
#hbar = const.hbar
#m_e = const.m_e
#seconds = 1

"""
Split-operator method for the Schrödinger equation.

Prototype and original implementation:
https://github.com/marl0ny/split-operator-simulations

References:
https://www.algorithm-archive.org/contents/
split-operator_method/split-operator_method.html
https://en.wikipedia.org/wiki/Split-step_method
"""

class NonlinearSplitStepMethod():

    """
    Split-Operator method for the non-linear Schrodinger equation.
    
    References:

      Xavier Antoine, Weizhu Bao, Christophe Besse
      Computational methods for the dynamics of 
      the nonlinear Schrodinger/Gross-Pitaevskii equations.
      Comput. Phys. Commun., Vol. 184, pp. 2621-2633, 2013.
      https://arxiv.org/pdf/1305.1093

    """

    def __init__(self, potential: np.ndarray,
                 dimensions: Tuple[float, ...],
                 timestep: Union[float, np.complex128] = 1e-5 * seconds,
                 m: float = m_e):
        if len(potential.shape) != len(dimensions):
            raise Exception('Potential shape does not match dimensions')
        self.m = m
        self.V = potential
        self._dim = dimensions
        self._exp_potential = None
        self._kinetic = None
        self._exp_kinetic = None
        self._norm = False
        self._dt = 0
        self._nonlinear = lambda psi: psi
        #self.set_timestep(timestep)

    def set_timestep(self, timestep: Union[float, np.complex128]) -> None:
        """
        Set the timestep. It can be real or complex.
        """
        self._dt = timestep
        self._exp_potential = np.exp(-0.5j*(self._dt/hbar)*self.V)
        p = np.meshgrid(*[2.0*np.pi*hbar*np.fft.fftfreq(d)*d/
                          self._dim[i] for i, d in enumerate(self.V.shape)])
        self._kinetic = sum([p_i**2 for p_i in p])/(2.0*self.m)
        self._exp_kinetic = np.exp(-1j*(self._dt/(2.0*self.m*hbar))
                                   * sum([p_i**2 for p_i in p]))

    def set_potential(self, V: np.ndarray) -> None:
        """
        Change the potential
        """
        self.V = V
        self._exp_potential = np.exp(-0.5j*(self._dt/hbar)*self.V)

    def __call__(self,psi,t, particle: np.ndarray) -> np.ndarray:
        """
        Step the wavefunction in time.
        """
        psi = self._nonlinear(psi,t,particle)
        psi_p = np.fft.fftn(psi*self._exp_potential)
        psi_p = psi_p*self._exp_kinetic
        psi = np.fft.ifftn(psi_p)*self._exp_potential
        psi = self._nonlinear(psi,t,particle)
        if self._norm:
            psi = psi/np.sqrt(np.sum(psi*np.conj(psi)))
        return psi

    def get_expected_energy(self, psi: np.ndarray) -> float:
        """
        Get the energy expectation value of the wavefunction
        """
        psi_p = np.fft.fftn(psi)
        psi_p = psi_p/np.sqrt(np.sum(psi_p*np.conj(psi_p)))
        kinetic = np.real(np.sum(np.conj(psi_p)*self._kinetic*psi_p))
        potential = np.real(np.sum(self.V*np.conj(psi)*psi))
        return kinetic + potential

    def normalize_at_each_step(self, norm: bool) -> None:
        """
        Whether to normalize the wavefunction at each time step or not.
        """
        self._norm = norm

    def set_nonlinear_term(self, nonlinear_func: Callable) -> None:
        """
        Set the nonlinear term.
        """
        self._nonlinear = lambda psi,t,particle: psi*np.exp(-0.5j*nonlinear_func(psi,t,particle)*self._dt/hbar)


    

class NonlinearSplitStep(Method):
    def __init__(self, simulation):

        self.simulation = simulation
        self.H = simulation.H
        self.simulation.Vmin = np.amin(self.H.Vgrid)
        self.simulation.Vmax = np.amax(self.H.Vgrid)
        #self.H.particle_system.compute_momentum_space(self.H)
        #self.p2 = self.H.particle_system.p2
        L = self.H.extent
        Z = self.H.z_extent
        if self.H.spatial_ndim == 1:
            self.split_step = NonlinearSplitStepMethod(self.H.potential(self.H.particle_system),(L,),1e-5 * seconds,m = self.H.particle_system.m)
        elif self.H.spatial_ndim == 2:
            self.split_step = NonlinearSplitStepMethod(self.H.potential(self.H.particle_system),(L,L,),1e-5 * seconds,m = self.H.particle_system.m)
        else:
            self.split_step = NonlinearSplitStepMethod(self.H.potential(self.H.particle_system),(L,L,Z),1e-5 * seconds,m = self.H.particle_system.m)


    def run(self, initial_wavefunction, total_time, dt, store_steps = 1):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import scipy.constants as const
        
        #print(dt)
        self.split_step.set_timestep(dt)
        self.simulation.store_steps = store_steps
        dt_store = total_time/store_steps
        self.simulation.total_time = total_time

        Nt = int(np.round(total_time / dt))
        Nt_per_store_step = int(np.round(dt_store / dt))
        self.simulation.Nt_per_store_step = Nt_per_store_step

        #time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
        self.simulation.dt = dt_store/Nt_per_store_step

        if isinstance(self.simulation.H.particle_system ,SingleParticle):
            Ψ = np.zeros((store_steps + 1, *([self.H.N] *self.H.ndim )), dtype = np.complex128)

        elif isinstance(self.simulation.H.particle_system,TwoParticles):
            Ψ = np.zeros((store_steps + 1, *([self.H.N] * 2)), dtype = np.complex128)

        Ψ[0] = np.array(initial_wavefunction(self.H.particle_system))
        
        #steps_image=int(total_time/dt/store_steps)  # Number of computational steps between consecutive graphic outputs

        # Main computational loop
        #print("calculating", end="", flush=True)
        
        t0 = time.time()
        bar = progressbar.ProgressBar()
        t_count = 0
        for i in bar(range(store_steps)):
            tmp = np.copy(Ψ[i])
            for j in range(Nt_per_store_step):
                t_count += 1
                t = t_count * self.simulation.dt
                tmp = self.split_step(tmp,t,self.simulation.H.particle_system)
            Ψ[i+1] = tmp
        print("Took", time.time() - t0)
        
        
        self.simulation.Ψ = Ψ
        self.simulation.Ψmax = np.amax(np.abs(Ψ))



import cupy as cp
class NonlinearSplitStepMethodCupy():

    """
    Split-Operator method for the non-linear Schrodinger equation.
    
    References:

      Xavier Antoine, Weizhu Bao, Christophe Besse
      Computational methods for the dynamics of 
      the nonlinear Schrodinger/Gross-Pitaevskii equations.
      Comput. Phys. Commun., Vol. 184, pp. 2621-2633, 2013.
      https://arxiv.org/pdf/1305.1093

    """
    
    def __init__(self, potential: np.ndarray,
                 dimensions: Tuple[float, ...],
                 timestep: Union[float, np.complex128] = 1e-5 * seconds,
                 m: float = m_e):
        if len(potential.shape) != len(dimensions):
            raise Exception('Potential shape does not match dimensions')
        self.m = m
        self.V = cp.array(potential)
        self._dim = dimensions
        self._exp_potential = None
        self._kinetic = None
        self._exp_kinetic = None
        self._norm = False
        self._dt = 0
        #self.set_timestep(timestep)
        self._nonlinear = lambda psi: psi

    def set_timestep(self, timestep: Union[float, np.complex128]) -> None:
        """
        Set the timestep. It can be real or complex.
        """
        self._dt = timestep
        self._exp_potential = cp.exp(-0.5j*(self._dt/hbar)*cp.array(self.V))
        p = np.meshgrid(*[2.0*np.pi*hbar*cp.fft.fftfreq(d)*d/
                          self._dim[i] for i, d in enumerate(self.V.shape)])
        self._kinetic = sum([p_i**2 for p_i in p])/(2.0*self.m)
        self._exp_kinetic = cp.exp(-1j*(self._dt/(2.0*self.m*hbar))
                                   * sum([p_i**2 for p_i in p]))

    def set_potential(self, V: np.ndarray) -> None:
        """
        Change the potential
        """
        self.V = V
        self._exp_potential = np.exp(-0.5j*(self._dt/hbar)*self.V)

    def __call__(self,psi,t, particle: np.ndarray) -> np.ndarray:
        """
        Step the wavefunction in time.
        """ 
        psi = cp.array(self._nonlinear(psi,t,particle))
        psi_p = cp.fft.fftn(psi*cp.array(self._exp_potential))
        psi_p = psi_p*self._exp_kinetic
        psi = cp.fft.ifftn(psi_p)*self._exp_potential
        psi = cp.array(self._nonlinear(psi,t,particle))
        if self._norm:
            psi = psi/cp.sqrt(np.sum(psi*np.conj(psi)))
        return psi


    def normalize_at_each_step(self, norm: bool) -> None:
        """
        Whether to normalize the wavefunction at each time step or not.
        """
        self._norm = norm
        

    def set_nonlinear_term(self, nonlinear_func: Callable) -> None:
        """
        Set the nonlinear term.
        """
        self._nonlinear = lambda psi,t,particle: psi*np.exp(-0.5j*nonlinear_func(psi,t,particle)*self._dt/hbar)

class NonlinearSplitStepCupy(Method):
    def __init__(self, simulation):

        self.simulation = simulation
        self.H = simulation.H
        self.simulation.Vmin = np.amin(self.H.Vgrid)
        self.simulation.Vmax = np.amax(self.H.Vgrid)

        #self.H.particle_system.compute_momentum_space(self.H)
        #self.p2 = self.H.particle_system.p2
        #self.split_step = NonlinearSplitStepMethodCupy(self.H.potential(self.H.particle_system),(self.H.extent/2,),1e-5 * seconds,m = self.H.particle_system.m)
        L = self.H.extent
        Z = self.H.z_extent
        if self.H.spatial_ndim == 1:
            self.split_step = NonlinearSplitStepMethodCupy(self.H.potential(self.H.particle_system),(L,),1e-5 * seconds,m = self.H.particle_system.m)
        elif self.H.spatial_ndim == 2:
            self.split_step = NonlinearSplitStepMethodCupy(self.H.potential(self.H.particle_system),(L,L,),1e-5 * seconds,m = self.H.particle_system.m)
        else:
            self.split_step = NonlinearSplitStepMethodCupy(self.H.potential(self.H.particle_system),(L,L,Z),1e-5 * seconds,m = self.H.particle_system.m)
        



    def run(self, initial_wavefunction, total_time, dt, store_steps = 1):

        import cupy as cp 
        #self.p2 = cp.array(self.p2)
        self.split_step.set_timestep(dt)
        self.simulation.store_steps = store_steps
        dt_store = total_time/store_steps
        self.simulation.total_time = total_time

        Nt = int(np.round(total_time / dt))
        Nt_per_store_step = int(np.round(dt_store / cp.abs(dt)))
        self.simulation.Nt_per_store_step = Nt_per_store_step

        #time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
        self.simulation.dt = dt_store/Nt_per_store_step
        

        #Ψ = cp.zeros((store_steps + 1, *([self.H.N] *self.H.ndim )), dtype = cp.complex128)
        Ψ = 0
        
        if self.H.ndim == 3:
            Ψ = cp.zeros((store_steps + 1, self.H.N,self.H.N, self.H.Nz), dtype = cp.complex128)
        else:
            Ψ = cp.zeros((store_steps + 1, *([self.H.N] *self.H.ndim )), dtype = cp.complex128)
        Ψ[0] = cp.array(initial_wavefunction(self.H.particle_system))
        


        #tmp = cp.copy(Ψ[0])
        
        #steps_image=int(total_time/dt/store_steps)  # Number of computational steps between consecutive graphic outputs

        # Main computational loop
        t0 = time.time()
        bar = progressbar.ProgressBar()
        t_count = 0
        for i in bar(range(store_steps)):
            tmp = cp.copy(Ψ[i])
            for j in range(Nt_per_store_step):
                t_count += 1
                t = t_count * self.simulation.dt
                tmp = self.split_step(tmp,t,self.simulation.H.particle_system)
            Ψ[i+1] = tmp
        print("Took", time.time() - t0)
    
        self.simulation.Ψ = Ψ.get()
        self.simulation.Ψmax = np.amax(np.abs(self.simulation.Ψ ))
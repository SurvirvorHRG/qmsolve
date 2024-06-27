"""
Single particle quantum mechanics simulation
using the split-operator method.

References:
https://www.algorithm-archive.org/contents/
split-operator_method/split-operator_method.html

https://en.wikipedia.org/wiki/Split-step_method

"""
import numpy as np
from ..util.constants import hbar, Å, femtoseconds
from ..hamiltonian import Hamiltonian
import time
import matplotlib.pyplot as plt
from ..util.colour_functions import complex_to_rgba

from .split_step import SplitStep, SplitStepCupy
from .nonlinear_splitstep import NonlinearSplitStep,NonlinearSplitStepCupy
from .crank_nicolson import CrankNicolson, CrankNicolsonCupy

class TimeSimulation:
    """
    Class for configuring time dependent simulations.
    """

    def __init__(self, hamiltonian, method = "split-step"):

        self.H = hamiltonian
        self.method_name = method

        implemented_solvers = ('split-step','nonlinear-split-step','nonlinear-split-step-cupy', 'split-step-cupy', 'crank-nicolson', 'crank-nicolson-cupy')

        if method == "split-step":

            if self.H.potential_type == "grid":
                self.method = SplitStep(self)
            else:
                raise NotImplementedError(
                f"split-step can only be used with grid potential_type. Use crank-nicolson instead")
                
        elif method == "nonlinear-split-step":
    
            if self.H.potential_type == "grid":
                self.method = NonlinearSplitStep(self)
            else:
                raise NotImplementedError(
                f"split-step can only be used with grid potential_type. Use crank-nicolson instead")

        elif method == "split-step-cupy":

            if self.H.potential_type == "grid":
                self.method = SplitStepCupy(self)
            else:
                raise NotImplementedError(
                f"split-step can only be used with grid potential_type. Use crank-nicolson instead")

        elif method == "nonlinear-split-step-cupy":
    
            if self.H.potential_type == "grid":
                self.method = NonlinearSplitStepCupy(self)
            else:
                raise NotImplementedError(
                f"split-step can only be used with grid potential_type. Use crank-nicolson instead")


        elif method == "crank-nicolson":
            self.method = CrankNicolson(self)
            
        elif method == "crank-nicolson-cupy":
            self.method = CrankNicolsonCupy(self)
        else:
            raise NotImplementedError(
                f"{method} solver has not been implemented. Use one of {implemented_solvers}")



    def run(self, initial_wavefunction, total_time, dt, store_steps = 1, non_linear_function = None,norm = False, g = 1):
        """
        """
        if self.method_name == "split-step" or self.method_name == "split-step-cupy":
            self.method.run(initial_wavefunction, total_time, dt, store_steps, non_linear_function,norm,g)
        else:
            self.method.run(initial_wavefunction, total_time, dt, store_steps)
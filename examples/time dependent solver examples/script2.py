import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class readFile:
    def __init__(self, filename):
        # Initialize variables to store loaded data
        self.Nx = None
        self.xmax = None
        self.total_time = None
        self.store_steps = None
        self.psi_norm = None
        
        # Load arrays and parameters from the text file
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Read parameters
            self.Nx = int(lines[0].split(": ")[1])
            self.xmax = float(lines[1].split(": ")[1].strip())
            self.total_time = float(lines[2].split(": ")[1].strip())
            self.store_steps = int(lines[3].split(": ")[1].strip())
            
            # Read arrays
            array1_start = lines.index("psi_norm:\n") + 1
            self.psi_norm = np.loadtxt(lines[array1_start:], delimiter=',')
            

    def final_plot2(self, ax, L_norm=1, Z_norm=1, unit=1, time="ms", fixmaximum=0):
        plt.style.use("default")
        total_time = self.total_time
        tvec = np.linspace(0, total_time, self.store_steps + 1)
        x = np.linspace(-self.xmax/2, self.xmax/2, self.Nx)
        tt, xx = np.meshgrid(tvec, x)
    
        toplot = np.abs(self.psi_norm)**2
        if fixmaximum > 0:
            toplot[toplot > fixmaximum] = fixmaximum
    
        cont = ax.contourf(xx / L_norm, tt / unit, toplot.T, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel('$z\ (\mu m)$', fontsize=44)  # axes labels, title, plot and axes range
        ax.set_ylabel('$t\ (ms)$', fontsize=44)
        ax.tick_params(axis='both', which='major', labelsize=40)  # Increase tick label size
        return cont


hbar = 1.054571596e-34
uaumass = 1.66053873e-27
mass=86.909  # Atoms mass Cs 132.905 , Rb 86.909 (united atomic unit of mass)
mass  = mass * uaumass
Ntot= 20e4
omega_rho = 2*np.pi*160
omega_z = 2*np.pi*6.8
z_t = np.sqrt(hbar/mass/omega_z) # 3e-6 meters

plt.style.use("default")
font = 44

# Create subplots

fig, axs = plt.subplots(1, 2, figsize=(30, 15))

# Lithium
rd_lithium = readFile('l=3.txt')
cont_lithium = rd_lithium.final_plot2(axs[0],  1e-6,Z_norm =1e-6,unit = omega_z*1e-3)
axs[0].set_title('$(a)\ \ell = 3$', fontsize=font)

# Sodium
mass_sodium = 22.9 * uaumass  # Sodium
r_t_sodium = np.sqrt(hbar / mass_sodium / omega_rho)  # 3e-6 meters

rd_sodium = readFile('l=6.txt')
cont_sodium = rd_sodium.final_plot2(axs[1],  1e-6,Z_norm =  1e-6,unit = omega_z*1e-3)
axs[1].set_title('$(b)\ \ell = 6$', fontsize=font)



# Adjust layout
plt.tight_layout()

# Add a single colorbar for all subplots
cbar = fig.colorbar(cont_sodium, ax=axs, orientation='vertical', fraction=0.05, pad=0.01)
cbar.set_label('$|\psi|^2$', fontsize=44)
cbar.ax.tick_params(labelsize=40)  # Increase colorbar tick label size

# Display the plot
plt.show()

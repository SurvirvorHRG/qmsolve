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
        ax.set_xlabel('$z\ (mm)$', fontsize=44)  # axes labels, title, plot and axes range
        ax.set_ylabel('$t\ (ms)$', fontsize=44)
        ax.tick_params(axis='both', which='major', labelsize=40)  # Increase tick label size
        return cont


hbar = 1.054571596e-34
uaumass = 1.66053873e-27
omega_rho = 1.0e3  # 1kHz

plt.style.use("default")
font = 44

# Create subplots
mass_lithium = 7.016004 * uaumass  # Lithium
r_t_lithium = np.sqrt(hbar / mass_lithium / omega_rho)  # 3e-6 meters

fig, axs = plt.subplots(1, 3, figsize=(50, 15))

# Lithium
rd_lithium = readFile('lithium.txt')
cont_lithium = rd_lithium.final_plot2(axs[0], L_norm=1/r_t_lithium*1e-3, Z_norm=1/r_t_lithium*1e-3, unit=omega_rho*1e-3)
axs[0].set_title('$(a)\ ^{7}Li$', fontsize=font)

# Sodium
mass_sodium = 22.9 * uaumass  # Sodium
r_t_sodium = np.sqrt(hbar / mass_sodium / omega_rho)  # 3e-6 meters

rd_sodium = readFile('sodium.txt')
cont_sodium = rd_sodium.final_plot2(axs[1], L_norm=1/r_t_sodium*1e-3, Z_norm=1/r_t_sodium*1e-3, unit=omega_rho*1e-3)
axs[1].set_title('$(b)\ ^{23}Na$', fontsize=font)

# Rubidium
mass_rubidium = 86.909 * uaumass  # Rubidium
r_t_rubidium = np.sqrt(hbar / mass_rubidium / omega_rho)  # 3e-6 meters

rd_rubidium = readFile('rubidium.txt')
cont_rubidium = rd_rubidium.final_plot2(axs[2], L_norm=1/r_t_rubidium*1e-3, Z_norm=1/r_t_rubidium*1e-3, unit=omega_rho*1e-3)
axs[2].set_title('$(c)\ ^{87}Rb$', fontsize=font)

# Adjust layout
plt.tight_layout()

# Add a single colorbar for all subplots
cbar = fig.colorbar(cont_rubidium, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label('$|\psi|^2$', fontsize=44)
cbar.ax.tick_params(labelsize=40)  # Increase colorbar tick label size

# Display the plot
plt.show()

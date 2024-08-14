import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
font = 44
subfont = 40

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
        ax.set_xlabel('$z\ (\mu m)$', fontsize=font)  # axes labels, title, plot and axes range
        ax.set_ylabel('$t\ (ms)$', fontsize=font)
        ax.tick_params(axis='both', which='major', labelsize=subfont)  # Increase tick label size
        return cont


hbar = 1.054571596e-34
uaumass = 1.66053873e-27
omega_rho = 1.0e3  # 1kHz
omega_z = 0.01 * omega_rho
plt.style.use("default")

# Create subplots
mass_lithium = 7.016004 * uaumass  # Lithium
L_z = np.sqrt(hbar/mass_lithium/omega_z)

fig, axs = plt.subplots(1, 3, figsize=(30, 16))

# Lithium
rd_lithium = readFile('k=0.25.txt')
cont_lithium = rd_lithium.final_plot2(axs[0],L_norm = 1/L_z * 1e-6,Z_norm = 1/L_z * 1e-6,unit = omega_z * 1e-3,fixmaximum = 0.1)
axs[0].set_title('$(a)\ U_{z} = 0.25$', fontsize=font)

# Sodium


rd_sodium = readFile('k=0.5.txt')
cont_sodium = rd_sodium.final_plot2(axs[1], L_norm = 1/L_z * 1e-6,Z_norm = 1/L_z * 1e-6,unit = omega_z * 1e-3,fixmaximum = 0.1)
axs[1].set_title('$(b)\ U_{z} = 0.5$', fontsize=font)

# Rubidium


rd_rubidium = readFile('k=0.75.txt')
cont_rubidium = rd_rubidium.final_plot2(axs[2],L_norm = 1/L_z * 1e-6,Z_norm = 1/L_z * 1e-6,unit = omega_z * 1e-3,fixmaximum = 0.1)
axs[2].set_title('$(c)\ U_{z} = 0.75$', fontsize=font)

"""
rdi = readFile('k=1.txt')
cont_rubidium = rdi.final_plot2(axs[3],L_norm = 1/L_z * 1e-6,Z_norm = 1/L_z * 1e-6,unit = omega_z * 1e-3,fixmaximum = 0.1)
axs[3].set_title('$(d)\ U_{z} = 1$', fontsize=font)
"""
# Adjust layout
plt.tight_layout()

# Add a single colorbar for all subplots
cbar = fig.colorbar(cont_rubidium, ax=axs, orientation='vertical', fraction=0.04, pad=0.01)
cbar.set_label('$|\psi|^2$', fontsize=font)
cbar.ax.tick_params(labelsize=subfont)  # Increase colorbar tick label size

# Display the plot
plt.show()

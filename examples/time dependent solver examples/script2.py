import numpy as np
import matplotlib.pyplot as plt

class readFile():
    def __init__(self,filename):
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
            

    def final_plot2(self,ax, L_norm=1, Z_norm=1, unit=1, time="ms", fixmaximum=0,n=0):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        plt.style.use("default")
        total_time = self.total_time
        tvec = np.linspace(0, total_time, self.store_steps + 1)
        x = np.linspace(-rd.xmax/2, rd.xmax/2, rd.Nx)
        tt, xx = np.meshgrid(tvec, x)
    
        toplot=np.abs(rd.psi_norm)**2
        if fixmaximum > 0:
            toplot[toplot > fixmaximum] = fixmaximum
    
        cont = ax.contourf(xx / L_norm, tt / unit, toplot.T, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        if n == 3:
            cbar = plt.colorbar(cont, ax=ax)  # colorbar
        ax.set_xlabel('$z\ (mm)$', fontsize=34)  # axes labels, title, plot and axes range
        ax.set_ylabel('$t\ (ms)$', fontsize=34)
        if n == 3:
            cbar.set_label('$|\psi|^2$', fontsize=34)
        ax.tick_params(axis='both', which='major', labelsize=30)  # increase tick label size
        if n == 3:
            cbar.ax.tick_params(labelsize=30)  # increase colorbar tick label size
            

hbar=1.054571596e-34
uaumass=1.66053873e-27     
omega_rho = 1.0e3 # 1kHz     
rd = readFile('lithium.txt')

plt.style.use("default")
font = 34

# Create subplots
#Lithium
mass=7.016004 * uaumass # Lithium
r_t = np.sqrt(hbar/mass/omega_rho) # 3e-6 meters

fig, axs = plt.subplots(1, 3, figsize=(50, 15))
# Plot on each subplot using the modified final_plot function
rd.final_plot2(axs[0],L_norm = 1/r_t*1e-3,Z_norm = 1/r_t*1e-3,unit = omega_rho*1e-3 )
axs[0].set_title('$^{7}Li$',fontsize = font)


# Sodium
mass=22.9 * uaumass # Sodium

rd = readFile('sodium.txt')
rd.final_plot2(axs[1],L_norm = 1/r_t*1e-3,Z_norm = 1/r_t*1e-3,unit = omega_rho*1e-3 )
axs[1].set_title('$^{23}Na$',fontsize = font)

# Rubidium

mass=86.909 * uaumass # Rubidium


rd = readFile('rubidium.txt')
rd.final_plot2(axs[2],L_norm = 1/r_t*1e-3,Z_norm = 1/r_t*1e-3,unit = omega_rho*1e-3,n=3)
axs[2].set_title('$^{87}Rb$',fontsize = font)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()
        

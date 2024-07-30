import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mayavi import mlab
from mlabtex import mlabtex

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
    
    def final_plot3(self, ax, L_norm=1, Z_norm=1, unit=1, time="ms", fixmaximum=0):
        plt.style.use("default")
        total_time = self.total_time
        tvec = np.linspace(0, total_time, self.store_steps + 1)
        x = np.linspace(-self.xmax / 2, self.xmax / 2, self.Nx)
        tt, xx = np.meshgrid(tvec, x)
    
        toplot = np.abs(self.psi_norm) ** 2
        if fixmaximum > 0:
            toplot[toplot > fixmaximum] = fixmaximum
    
        ax.plot_surface(xx / L_norm, tt / unit, toplot.T, cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel('$z\ (\mu m)$', fontsize=44)
        ax.set_ylabel('$t\ (ms)$', fontsize=44)
        ax.set_zlabel('$|\psi|^2$', fontsize=44)
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.tick_params(axis='z', which='major', labelsize=40)  # Increase z-axis tick label size
        return ax
    
    def final_plot3D(self,L_norm = 1, Z_norm = 1,unit = 1):
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        
        total_time = self.total_time
        tvec = np.linspace(0, total_time, self.store_steps + 1)
        x = np.linspace(-self.xmax / 2, self.xmax / 2, self.Nx)
        tt, xx = np.meshgrid(tvec, x)
        toplot = np.abs(self.psi_norm) ** 2

        toplot = toplot.T
        """
        max_val = np.max(toplot)
        min_val = np.min(toplot)
        
        # Reverse the values
        toplot = max_val - (toplot - min_val)
        """
        mlab.figure(fgcolor=(0.1,0.1,0.1),bgcolor=(1,1,1),size=(700, 700))
        L = self.xmax/2/L_norm
        N = self.Nx
        
        #surf = mlab.mesh(zz/Z_norm/Å,tt/unit,toplot,colormap='jet')
        surf = mlab.surf(toplot,warp_scale="auto",colormap='jet')
        #surf.module_manager.scalar_lut_manager.reverse_lut = True
        #surf = mlab.surf(toplot,warp_scale="auto",colormap='jet')
        
        #surf = mlab.surf(psi[:,:,49],colormap='jet')
        #mlab.colorbar(surf,title='psi',orientation='vertical')  
              

        #mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=3 , ranges = (-L,L,0,total_time/unit,np.min(toplot),np.max(toplot)) )
        ax = mlab.axes(xlabel='$z\ (mm)$', ylabel='$t\ (s)$', zlabel='',nb_labels=3 , ranges = (-L,L,0,total_time/unit,np.min(toplot),np.max(toplot)) )
        #ax.axes.y_axis_visibility = False
        ax.axes.font_factor = 1.3
        ax.label_text_property.font_family = 'times'
        ax.title_text_property.font_family = 'times'
        ax.axes.label_format = '%-#6.1g'
        #colorbar = mlab.colorbar(orientation = 'vertical')
        #colorbar.scalar_bar_representation.position = [0.85, 0.1]
        mlab.show()
        
    def final_plot_3D_x(self,L_norm = 1, Z_norm = 1,unit = 1, figsize=(15, 15),time="ms"):
        
        from mpl_toolkits.mplot3d import Axes3D
        rd =self
        x = np.linspace(-rd.xmax/2, rd.xmax/2, rd.Nx)
        y = np.linspace(-rd.xmax/2, rd.xmax/2, rd.Nx)
        xx,yy=np.meshgrid(x,y)
        z = xx[0,:]
        total_time = rd.total_time
        tvec=np.linspace(0,total_time,rd.store_steps+1)
        tt,zz=np.meshgrid(tvec,z)

        
        # Generates the plot
        mid = int(rd.Nx / 2) - 1
        toplot= rd.psi_norm[:,mid,:]
        toplot = toplot.T
        
        mlab.figure(fgcolor=(0.1,0.1,0.1),bgcolor=(1,1,1),size=(700, 700))
        L = self.xmax/2/L_norm
        N = self.Nx
        
        #surf = mlab.mesh(zz/Z_norm/Å,tt/unit,toplot,colormap='jet')
        surf = mlab.surf(toplot,warp_scale="auto",colormap='jet')
        #surf.module_manager.scalar_lut_manager.reverse_lut = True
        #surf = mlab.surf(toplot,warp_scale="auto",colormap='jet')
        
        #surf = mlab.surf(psi[:,:,49],colormap='jet')
        #mlab.colorbar(surf,title='psi',orientation='vertical')  
              

        #mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=3 , ranges = (-L,L,0,total_time/unit,np.min(toplot),np.max(toplot)) )
        ax = mlab.axes(xlabel='$z\ (mm)$', ylabel='$t\ (s)$', zlabel='',nb_labels=3 , ranges = (-L,L,0,total_time/unit,np.min(toplot),np.max(toplot)) )
        #ax.axes.y_axis_visibility = False
        ax.axes.font_factor = 1.3
        ax.label_text_property.font_family = 'times'
        ax.title_text_property.font_family = 'times'
        ax.axes.label_format = '%-#6.1g'
        #colorbar = mlab.colorbar(orientation = 'vertical')
        #colorbar.scalar_b
        mlab.show()


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

#rd_lithium.final_plot3D(L_norm = 1e-3,unit = 1)
rd_lithium.final_plot_3D_x(L_norm = 1e-3,unit = 1)
#rd_sodium.final_plot3D(L_norm = 1e-3,unit = 1)

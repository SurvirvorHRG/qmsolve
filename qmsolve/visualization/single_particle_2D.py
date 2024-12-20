import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import animation
from .visualization import Visualization
from ..util.colour_functions import complex_to_rgb, complex_to_rgba
from ..util.constants import *
from mayavi import mlab

class VisualizationSingleParticle2D(Visualization):
    def __init__(self,eigenstates):
        self.eigenstates = eigenstates



    def plot_eigenstate(self, k, xlim=None, ylim=None):
        eigenstates_array = self.eigenstates.array
        energies = self.eigenstates.energies

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(16/9 *5.804 * 0.9,5.804)) 

        grid = plt.GridSpec(2, 2, width_ratios=[4.5, 1], height_ratios=[1, 1] , hspace=0.1, wspace=0.2)
        ax1 = fig.add_subplot(grid[0:2, 0:1])
        ax2 = fig.add_subplot(grid[0:2, 1:2])

        ax1.set_xlabel("$x$ [Å]")
        ax1.set_ylabel("$y$ [Å]")
        ax1.set_title("$\Psi(x,y)$")

        ax2.set_title('Energy Level')
        ax2.set_facecolor('black')
        ax2.set_ylabel('$E_N$ [eV]')
        ax2.set_xticks(ticks=[])

        if xlim != None:
            ax1.set_xlim(np.array(xlim)/Å)
        if ylim != None:
            ax1.set_ylim(np.array(ylim)/Å)

        E0 = energies[0]

        for E in energies:
            ax2.plot([0,1], [E, E], color='gray', alpha=0.5)

        ax2.plot([0,1], [energies[k], energies[k]], color='yellow', lw = 3)

        ax1.set_aspect('equal')
        L =  self.eigenstates.extent/2/Å
        im = ax1.imshow(complex_to_rgb(eigenstates_array[k]*np.exp( 1j*2*np.pi/10*k)), origin='lower',extent = [-L, L, -L, L],  interpolation = 'bilinear')
        plt.show()


    def slider_plot(self, xlim=None, ylim=None):

        eigenstates_array = self.eigenstates.array
        energies = self.eigenstates.energies

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(16/9 *5.804 * 0.9,5.804)) 

        grid = plt.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 1] , hspace=0.1, wspace=0.2)
        ax1 = fig.add_subplot(grid[0:2, 0:1])
        ax2 = fig.add_subplot(grid[0:2, 1:2])

        ax1.set_xlabel("$x$ [Å]")
        ax1.set_ylabel("$y$ [Å]")
        ax1.set_title("$\Psi(x,y)$")

        ax2.set_title('Energy Level')
        ax2.set_facecolor('black')
        ax2.set_ylabel('$E_N$ [eV]')
        ax2.set_xticks(ticks=[])

        if xlim != None:
            ax1.set_xlim(np.array(xlim)/Å)
        if ylim != None:
            ax1.set_ylim(np.array(ylim)/Å)


        E0 = energies[0]
        for E in energies:
            ax2.plot([0,1], [E, E], color='gray', alpha=0.5)


        ax1.set_aspect('equal')
        L = self.eigenstates.extent/2/Å
        eigenstate_plot = ax1.imshow(complex_to_rgb(eigenstates_array[0]*np.exp( 1j*2*np.pi/10*0)), origin='lower',extent = [-L, L, -L, L],  interpolation = 'bilinear')
        
        line = ax2.plot([0,1], [energies[0], energies[0]], color='yellow', lw = 3)

        plt.subplots_adjust(bottom=0.2)
        from matplotlib.widgets import Slider
        slider_ax = plt.axes([0.2, 0.05, 0.7, 0.05])
        slider = Slider(slider_ax,      # the axes object containing the slider
                          'state',            # the name of the slider parameter
                          0,          # minimal value of the parameter
                          len(eigenstates_array)-1,          # maximal value of the parameter
                          valinit = 0,  # initial value of the parameter 
                          valstep = 1,
                          color = '#5c05ff' 
                         )

        def update(state):
            state = int(state)
            eigenstate_plot.set_data(complex_to_rgb(eigenstates_array[state]*np.exp( 1j*2*np.pi/10*state)))
            line[0].set_ydata([energies[state], energies[state]])

        slider.on_changed(update)
        plt.show()






    def animate(self,seconds_per_eigenstate = 0.5, fps = 20, max_states = None, xlim=None, ylim=None, save_animation = False):

        if max_states == None:
            max_states = len(self.eigenstates.energies)

        frames_per_eigenstate = fps * seconds_per_eigenstate
        total_time = max_states * seconds_per_eigenstate
        total_frames = int(fps * total_time)


        eigenstates_array = self.eigenstates.array
        energies = self.eigenstates.energies

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(16/9 *5.804 * 0.9,5.804)) 

        grid = plt.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 1] , hspace=0.1, wspace=0.2)
        ax1 = fig.add_subplot(grid[0:2, 0:1])
        ax2 = fig.add_subplot(grid[0:2, 1:2])

        ax1.set_xlabel("$x$ [Å]")
        ax1.set_ylabel("$y$ [Å]")
        ax1.set_title("$\Psi(x,y)$")

        ax2.set_title('Energy Level')
        ax2.set_facecolor('black')
        ax2.set_ylabel('$E_N$ [eV]')
        ax2.set_xticks(ticks=[])

        if xlim != None:
            ax1.set_xlim(np.array(xlim)/Å)
        if ylim != None:
            ax1.set_ylim(np.array(ylim)/Å)

        E0 = energies[0]
        for E in energies:
            ax2.plot([0,1], [E, E], color='gray', alpha=0.5)
        
        ax1.set_aspect('equal')
        L = self.eigenstates.extent/2/Å
        eigenstate_plot = ax1.imshow(complex_to_rgb(eigenstates_array[0]*np.exp( 1j*2*np.pi/10*0)),  origin='lower',extent = [-L, L, -L, L],   interpolation = 'bilinear')

        line, = ax2.plot([0,1], [energies[0], energies[0]], color='yellow', lw = 3)

        plt.subplots_adjust(bottom=0.2)

        import matplotlib.animation as animation

        animation_data = {'n': 0.0}
        Δn = 1/frames_per_eigenstate

        def func_animation(*arg):
            animation_data['n'] = (animation_data['n'] + Δn) % len(energies)
            state = int(animation_data['n'])
            if (animation_data['n'] % 1.0) > 0.5:
                transition_time = (animation_data['n'] - int(animation_data['n']) - 0.5)
                eigenstate_combination = (np.cos(np.pi*transition_time)*eigenstates_array[state]*np.exp( 1j*2*np.pi/10*state) + 
                                         np.sin(np.pi*transition_time)*
                                         eigenstates_array[(state + 1) % len(energies)]*np.exp( 1j*2*np.pi/10*(state + 1)) )
                
                eigenstate_plot.set_data(complex_to_rgb(eigenstate_combination))


                E_N = energies[state] 
                E_M = energies[(state + 1) % len(energies)]
                E =  E_N*np.cos(np.pi*transition_time)**2 + E_M*np.sin(np.pi*transition_time)**2
                line.set_ydata([E, E])
            else:
                line.set_ydata([energies[state], energies[state]])
                eigenstate_combination = eigenstates_array[int(state)]*np.exp( 1j*2*np.pi/10*state)
                eigenstate_plot.set_data(complex_to_rgb(eigenstate_combination))
            return eigenstate_plot, line

        a = animation.FuncAnimation(fig, func_animation,
                                    blit=True, frames=total_frames, interval= 1/fps * 1000)
        if save_animation == True:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            a.save('animation.mp4', writer=writer)
        else:
            plt.show()


    def superpositions(self, states, fps = 30, total_time = 20, **kw):
        params = {'dt': 0.001, 'xlim': [-self.eigenstates.extent/2, 
                                        self.eigenstates.extent/2],
                  'ylim': [-self.eigenstates.extent/2, 
                           self.eigenstates.extent/2],
                  'save_animation': False,
                  'hide_controls': False,
                  # 'plot_style': 'dark_background'
                 }
        for k in kw.keys():
            params[k] = kw[k]
        total_frames = fps * total_time
        from .complex_slider_widget import ComplexSliderWidget
        eigenstates = self.eigenstates.array
        energies = self.eigenstates.energies
        eigenstates = np.array(eigenstates)
        energies = np.array(energies)
        coeffs = None
        if isinstance(states, int) or isinstance(states, float):
            coeffs = np.array([1.0 if i == 0 else 0.0 for i in range(states)],
                           dtype=np.complex128)
            eigenstates = eigenstates[0: states]
        else:
            coeffs = states
            eigenstates = eigenstates[0: len(states)]
            states = len(states)
            params[k] = kw[k]
        N = eigenstates.shape[1]
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(16/9 *5.804 * 0.9,5.804))
        grid_width = 10
        grid_length = states if states < 30 else 30
        grid = plt.GridSpec(grid_width, grid_length)
        grid_slice = grid[0:int(0.7*grid_width), 0:grid_length]
        if params['hide_controls']:
            grid_slice = grid[0:grid_width, 0:grid_length]
        ax = fig.add_subplot(grid_slice)
        ax.set_title("$\psi(x, y)$")
        ax.set_xlabel("$x$ [Å]")
        ax.set_ylabel("$y$ [Å]")
        # ax.set_xticks([])
        # ax.set_yticks([])
        get_norm_factor = lambda psi: 1.0/np.sqrt(np.sum(psi*np.conj(psi)))
        coeffs = np.array(coeffs, dtype=np.complex128)
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, eigenstates[0].shape[0]),
                        np.linspace(-1.0, 1.0, eigenstates[0].shape[1]))
        maxval = np.amax(np.abs(eigenstates[0]))


        ax.set_xlim(np.array(params['xlim'])/Å)
        ax.set_ylim(np.array(params['ylim'])/Å)


        im = plt.imshow(complex_to_rgb(eigenstates[0]), interpolation='bilinear',
                        origin='lower', extent=[-self.eigenstates.extent/2/Å, 
                                                self.eigenstates.extent/2/Å,
                                                -self.eigenstates.extent/2/Å, 
                                                self.eigenstates.extent/2/Å]
                        )
        # im2 = plt.imshow(0.0*eigenstates[0], cmap='gray')
        animation_data = {'ticks': 0, 'norm': 1.0}

        def make_update(n):
            def update(phi, r):
                coeffs[n] = r*np.exp(1.0j*phi)
                psi = np.dot(coeffs, 
                             eigenstates.reshape([states, N*N]))
                psi = psi.reshape([N, N])
                animation_data['norm'] = get_norm_factor(psi)
                psi *= animation_data['norm']
                # apsi = np.abs(psi)
                # im.set_alpha(apsi/np.amax(apsi))
            return update

        widgets = []
        circle_artists = []
        if not params['hide_controls']:
            for i in range(states):
                if states <= 30:
                    circle_ax = fig.add_subplot(grid[8:10, i], projection='polar')
                    circle_ax.set_title('n=' + str(i) # + '\nE=' + str() + '$E_0$'
                                        , size=8.0 if states < 15 else 6.0 
                                        )
                else:
                    circle_ax = fig.add_subplot(grid[8 if i < 30 else 9,
                                                     i if i < 30 else i-30], 
                                                projection='polar')
                    circle_ax.set_title('n=' + str(i) # + '\nE=' + str() + '$E_0$'
                                        , size=8.0 if states < 15 else 6.0 
                                        )
                circle_ax.set_xticks([])
                circle_ax.set_yticks([])
                widgets.append(ComplexSliderWidget(circle_ax, 0.0, 1.0, animated=True))
                widgets[i].on_changed(make_update(i))
                circle_artists.append(widgets[i].get_artist())
        artists = circle_artists + [im]

        def func(*args):
            animation_data['ticks'] += 1
            e = np.exp(-1.0j*energies[0:states]*params['dt'])
            np.copyto(coeffs, coeffs*e)
            norm_factor = animation_data['norm']
            psi = np.dot(coeffs*norm_factor, 
                         eigenstates.reshape([
                            states, N*N]))
            psi = psi.reshape([N, N])
            im.set_data(complex_to_rgb(psi))
            # apsi = np.abs(psi)
            # im.set_alpha(apsi/np.amax(apsi))
            # if animation_data['ticks'] % 2:
            #     return (im, )
            # else:
            if not params['hide_controls']:
                for i, c in enumerate(coeffs):
                    phi, r = np.angle(c), np.abs(c)
                    artists[i].set_xdata([phi, phi])
                    artists[i].set_ydata([0.0, r])
            return artists

        a = animation.FuncAnimation(fig, func, blit=True, interval= 1/fps * 1000,
                                    frames=None if (not params['save_animation']) else
                                    total_frames)
        if params['save_animation'] == True:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), 
                            bitrate=-1)
            a.save('animation.mp4', writer=writer)
            return
        plt.show()
        plt.show()



from .visualization import TimeVisualization

class TimeVisualizationSingleParticle2D(TimeVisualization):
    def __init__(self,simulation):
        self.simulation = simulation
        self.H = simulation.H
        
        
    def ax_plot(self, ax, t = 0, L_norm=1, Z_norm=1, unit=1, time="ms", fixmaximum=0):
        plt.style.use("default")
        x = np.linspace(-self.xmax/2, self.xmax/2, self.Nx)
        y = np.linspace(-self.xmax/2, self.xmax/2, self.Nx)
        x,y = np.meshgrid(x,y)

    
        # Makes the contour plot:
        index = int((self.store_steps)/self.total_time*t)
        toplot=self.psi_norm[index]    
        from matplotlib import cm
        cont = ax.contourf(x/L_norm, y/L_norm, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel("$x\ (mm)$",fontsize = 44)               # choose axes labels
        ax.set_ylabel("$y\ (mm)$",fontsize = 44)
        cbar.set_label('$|\psi|^2$',fontsize = 44)
        ax.tick_params(axis='both', which='major', labelsize=40)  # Increase tick label size
        cbar.ax.tick_params(labelsize=40)
        
    def save(self,filename):
        self.simulation.Ψ_plot = self.simulation.Ψ / self.simulation.Ψmax
        # Define some parameters
        Nx = self.simulation.H.N
        xmax = self.simulation.H.extent
        total_time = self.simulation.Nt_per_store_step * self.simulation.store_steps * self.simulation.dt
        store_steps = self.simulation.store_steps
        toplot = np.abs(self.simulation.Ψ_plot)
        # Save arrays and parameters to a text file
        with open(filename, 'w') as f:
            # Save parameters
            f.write(f"Nx: {Nx}\n")
            f.write(f"xmax: {xmax}\n")
            f.write(f"total_time: {total_time}\n")
            f.write(f"store_steps: {store_steps}\n")
            # Save shape of the 3D array
            f.write(f"psi_shape: {toplot.shape}\n")
            
            # Save psi
            f.write("psi_norm:\n")
            np.savetxt(f, toplot.reshape(-1, toplot.shape[-1]), delimiter=',')
            
            # Save array2
            #f.write("array2:\n")
            #np.savetxt(f, array2, delimiter=',')
            
            
    def save2(self,filename):
        self.simulation.Ψ_plot = self.simulation.Ψ / self.simulation.Ψmax
        # Define some parameters
        Nx = self.simulation.H.N
        xmax = self.simulation.H.extent
        total_time = self.simulation.Nt_per_store_step * self.simulation.store_steps * self.simulation.dt
        store_steps = self.simulation.store_steps
        toplot = np.abs(self.simulation.Ψ_plot)
        # Save arrays and parameters to a text file
        with open(filename, 'w') as f:
            # Save parameters
            f.write(f"Nx: {Nx}\n")
            f.write(f"xmax: {xmax}\n")
            f.write(f"total_time: {total_time}\n")
            f.write(f"store_steps: {store_steps}\n")
            
            # Save psi
            f.write("psi_norm:\n")
            np.savez(filename, array1=toplot,param1=Nx, param2=xmax,param3=total_time,param4=store_steps)
            
            # Save array2
            #f.write("array2:\n")
            #np.savetxt(f, array2, delimiter=',')        
            
    
        
    def final_plot_rho_t(self,L_norm = meters, Z_norm = meters,unit = milliseconds,time="ms",fixmaximum = 0):
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        plt.style.use("default")
        H = self.simulation.H
        total_time = self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt
        tvec=np.linspace(0,self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt,self.simulation.store_steps + 1)
        x = self.simulation.H.particle_system.x
        y = self.simulation.H.particle_system.y
        rho = np.sqrt(x**2 + y**2)
        tt,xx = np.meshgrid(tvec,rho)
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        toplot = abs(self.simulation.Ψ_plot).T
        from matplotlib import cm
        plt.contourf(xx/Z_norm, tt/unit, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)

        cbar=plt.colorbar()               # colorbar
        
        plt.xlabel('$rho$')               # choose axes labels, title of the plot and axes range
        plt.ylabel('$t\ (ms)$')
        cbar.set_label('$|\psi|^2$',fontsize=14)
        plt.show()      # Displays figure on screen
        
    def final_plot_xyt(self,L_norm = meters, Z_norm = meters,unit = milliseconds,time="ms",fixmaximum = 0):
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        plt.style.use("default")
        H = self.simulation.H
        total_time = self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt
        tvec=np.linspace(0,self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt,self.simulation.store_steps + 1)
        x = np.linspace(-H.extent/2, H.extent/2, H.N)
        y = np.linspace(-H.extent/2, H.extent/2, H.N)
        xx,yy,tt=np.meshgrid(x,y,tvec)
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        toplot = abs(self.simulation.Ψ_plot).T
        
        #Three-dimensional countour map
        fig = plt.figure(0,figsize=(9,9))
        ax = fig.add_subplot(projection=Axes3D.name)
        #threshold = 0  # Adjust the threshold as needed
        #high_values_mask = toplot >= threshold
        #toplot = toplot[high_values_mask]
    
        plot = ax.scatter3D(xx, yy,tt, toplot, c=toplot,cmap=cm.jet,antialiased=False)
        cbar = fig.colorbar(plot, shrink=0.5, aspect=5)
        ax.set_xlabel(r'X')
        ax.set_ylabel(r'Y')
        ax.set_zlabel(r'T')
        
        #cbar.set_label('$V$',fontsize=14)
        #ax.set_title('$l=$ %d'%(l))    # Title of the plot 
        
        cbar.set_label('$|\psi|^2$',fontsize=14)
        #ax.set_title('$t=$ %f'%(t))    # Title of the plot 
        
    def final_plot_xyt3(self,L_norm = 1, Z_norm = 1,unit = 1, contrast_vals= [0.1, 0.25]):

        mlab.figure(1,bgcolor=(0,0,0), size=(700, 700))
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        #index = int((self.simulation.store_steps)/self.simulation.total_time*t)   

            
        psi = self.simulation.Ψ_plot
        total_time = self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt
         
    
        L = self.simulation.H.extent/2/L_norm
        Z = self.simulation.H.z_extent/2/Z_norm
        N = self.simulation.H.N
        Nz = self.simulation.H.Nz
    
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.abs(psi).T), vmin= contrast_vals[0], vmax= contrast_vals[1])          
         
        mlab.outline()
         
         
     
        #mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
        ax = mlab.axes(xlabel='$y\ (mm)$', ylabel='$x\ (mm)$', zlabel='$t\ (ms)$',nb_labels=3 , ranges = (-L,L,-L,L,0,total_time) )
        ax.axes.font_factor = 1.3
        ax.label_text_property.font_family = 'times'
        ax.title_text_property.font_family = 'times'
        ax.axes.label_format = '%-#6.1g'
        
        colorbar = mlab.colorbar(nb_labels=6,orientation = 'vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        colorbar_label = colorbar.scalar_bar.label_text_property
        colorbar_label.font_family = 'times'
        colorbar.scalar_bar.label_format = '%.2f'
        colorbar.scalar_bar.unconstrained_font_size = True
        colorbar_label.font_size = 10
        mlab.view(azimuth=60,elevation=60,distance=(N + N + Nz)/3 * 4)
        colorbar = mlab.colorbar(orientation = 'vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        #colorbar.data_range = (0, 1)
        #file = str(t) + '.png'
        #mlab.savefig(file)
        mlab.show()
             
        
    def final_plot_x(self,L_norm = 1, Z_norm = 1,unit = 1, figsize=(15, 15),time="ms",fixmaximum = 0,fixminimum=0):
        
        from mpl_toolkits.mplot3d import Axes3D
        
        z = self.simulation.H.particle_system.x[0,:]
        tvec=np.linspace(0,self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt,self.simulation.store_steps+1)
        tt,zz=np.meshgrid(tvec,z)
        plt.figure("Evolution of 1D cut at y=0")              # figure
        plt.clf()                       # clears the figure
        
        # Generates the plot
        
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mid = int(self.simulation.H.N / 2) - 1
        toplot= np.abs(self.simulation.Ψ_plot[:,mid,:])
        toplot = toplot.T
        
        if fixminimum>0:
            toplot[toplot<fixminimum]=0
            
        if fixmaximum>0:
            toplot[toplot>fixmaximum]=fixmaximum
        
        from matplotlib import cm
        plt.contourf(zz/Z_norm, tt/unit, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        L = self.simulation.H.extent/2/L_norm
        Z = self.simulation.H.z_extent/2/Z_norm

        cbar=plt.colorbar()               # colorbar
        
        plt.xlabel('$x$')               # choose axes labels, title of the plot and axes range
        plt.ylabel('$t\ (s)$')
        cbar.set_label('$|\psi|^2$',fontsize=14)
        plt.show()      # Displays figure on screen
        
    def final_plot_y(self,L_norm = 1, Z_norm = 1,unit = 1, figsize=(15, 15),time="ms",fixmaximum = 0,fixminimum=0):
        
        from mpl_toolkits.mplot3d import Axes3D
        
        z = self.simulation.H.particle_system.y[:,0]
        tvec=np.linspace(0,self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt,self.simulation.store_steps+1)
        tt,zz=np.meshgrid(tvec,z)
        plt.figure("Evolution of 1D cut at y=0")              # figure
        plt.clf()                       # clears the figure
        
        # Generates the plot
        
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mid = int(self.simulation.H.N / 2) - 1
        toplot= np.abs(self.simulation.Ψ_plot[:,:,mid])
        toplot = toplot.T
        
        if fixminimum>0:
            toplot[toplot<fixminimum]=0
        
        from matplotlib import cm
        plt.contourf(zz/Z_norm, tt/unit, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        L = self.simulation.H.extent/2/L_norm
        Z = self.simulation.H.z_extent/2/Z_norm

        cbar=plt.colorbar()               # colorbar
        
        plt.xlabel('$y$')               # choose axes labels, title of the plot and axes range
        plt.ylabel('$t\ (s)$')
        cbar.set_label('$|\psi|^2$',fontsize=14)
        plt.show()      # Displays figure on screen


    def final_plot_3D_x(self,L_norm = 1, Z_norm = 1,unit = 1,extent=0,fixmaximum = 0):
        
        from mpl_toolkits.mplot3d import Axes3D
        L = self.simulation.H.extent/2/L_norm
        xmax = self.simulation.H.extent
        Nx = self.simulation.H.N
        x = np.linspace(-xmax/2, xmax/2, Nx)
        y = np.linspace(-xmax/2, xmax/2, Nx)
        xx,yy=np.meshgrid(x,y)
        z = xx[0,:]
        total_time = self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt
        tvec=np.linspace(0,total_time,self.simulation.store_steps+1)
        tt,zz=np.meshgrid(tvec,z)

        
        # Generates the plot
        if extent == 0:
            extent = self.simulation.H.extent

        L = extent/2/L_norm
        #extent = 10
        x_indices = np.where((x >= -extent/2) & (x <= extent/2))[0]
        y_indices = np.where((y >= -extent/2) & (y <= extent/2))[0]

        # Extract the submatrices for X, Y, and Z
        #X_sub = X[np.ix_(y_indices, x_indices)]
        #Y_sub = Y[np.ix_(y_indices, x_indices)]
        #Z_sub = Z[np.ix_(y_indices, x_indices)]
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mid = int(Nx / 2) - 1
        toplot= np.abs(self.simulation.Ψ_plot[:,mid,:])**2
        toplot = toplot[:, x_indices]
        toplot = toplot.T
        
        if fixmaximum>0:
            toplot[toplot>fixmaximum]=fixmaximum
        
        mlab.figure(fgcolor=(0.1,0.1,0.1),bgcolor=(1,1,1),size=(700, 700))

        
        #surf = mlab.mesh(zz/Z_norm/Å,tt/unit,toplot,colormap='jet')
        surf = mlab.surf(zz/L_norm,tt/unit,toplot,warp_scale='auto',colormap='jet')
        ax = mlab.axes(xlabel='$x\ (\mu m)$', ylabel='$t\ (ms)$', zlabel='',nb_labels=8 , ranges = (-L,L,0,total_time/unit,np.min(toplot),np.max(toplot)) )
        
        axes_actor = ax.axes
        #x_axis = axes_actor.label_text_property[0]
        ax.axes.font_factor = 0.9
        ax.label_text_property.font_family = 'times'
        ax.title_text_property.font_family = 'times'
        ax.axes.label_format = '%-#6.2g'
        colorbar = mlab.colorbar(nb_labels=6,orientation = 'vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        colorbar_label = colorbar.scalar_bar.label_text_property
        colorbar_label.font_family = 'times'
        colorbar.scalar_bar.unconstrained_font_size = True
        colorbar.scalar_bar.label_format = '%.2f'
        colorbar_label.font_size = 22
        #colorbar.scalar_b
        
        mlab.show()
        
        
    def plotSI(self, t,L_norm = 1,Z_norm = 1, figsize=(50, 50)):


        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("default")
        
        #Two-dimensional countour map
        fig = plt.figure("Contour map")    # figure
        plt.clf()                       # clears the figure
        fig.set_size_inches(8,6)
        plt.axes().set_aspect('equal')  # Axes aspect ratio
        
        plt.xlabel("$x$",fontsize = 14)               # choose axes labels
        plt.ylabel("$y$",fontsize = 14)
    
        # Makes the contour plot:
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)
        toplot=abs(self.simulation.Ψ_plot[index])     
        from matplotlib import cm
        plt.contourf(self.simulation.H.particle_system.x/L_norm, self.simulation.H.particle_system.y/L_norm, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        cbar=plt.colorbar()          # colorbar
    
        
        cbar.set_label('$|\psi|$',fontsize=14)
        plt.title('$t= %.2f\ (s)$' % (t))    # Title of the plot
        plt.show()
        

    def plot(self, t, xlim=None, ylim=None, figsize=(50, 50), potential_saturation=0.8, wavefunction_saturation=1.0):


        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("dark_background")

        fig = plt.figure(figsize=figsize)
        
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel("[Å]")
        ax.set_ylabel("[Å]")
        ax.set_title("$\psi(x,y,t)$")

        time_ax = ax.text(0.97,0.97, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top")
        time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (t/femtoseconds)))



        if xlim != None:
            ax.set_xlim(np.array(xlim)/Å)
        if ylim != None:
            ax.set_ylim(np.array(ylim)/Å)


        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)
        
        L = self.simulation.H.extent/Å
        ax.imshow((self.simulation.H.Vgrid + self.simulation.Vmin)/(self.simulation.Vmax-self.simulation.Vmin), vmax = 1.0/potential_saturation, vmin = 0, cmap = "gray", origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  

        ax.imshow(complex_to_rgba(self.simulation.Ψ_plot[index], max_val= wavefunction_saturation), origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  
        plt.show()


    def animate(self,xlim=None, ylim=None, figsize=(7, 7), animation_duration = 5, fps = 20, save_animation = False, potential_saturation=0.8, wavefunction_saturation=0.8):
        total_frames = int(fps * animation_duration)
        dt = self.simulation.total_time/total_frames
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("dark_background")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        index = 0
        
        L = self.simulation.H.extent/Å
        potential_plot = ax.imshow((self.simulation.H.Vgrid + self.simulation.Vmin)/(self.simulation.Vmax-self.simulation.Vmin), vmax = 1.0/potential_saturation, vmin = 0, cmap = "gray", origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  
        wavefunction_plot = ax.imshow(complex_to_rgba(self.simulation.Ψ_plot[0], max_val= wavefunction_saturation), origin = "lower", interpolation = "bilinear", extent=[-L / 2,L / 2,-L / 2,L / 2])


        if xlim != None:
            ax.set_xlim(np.array(xlim)/Å)
        if ylim != None:
            ax.set_ylim(np.array(ylim)/Å)

        import matplotlib.animation as animation

        ax.set_title("$\psi(x,y,t)$")
        ax.set_xlabel('[Å]')
        ax.set_ylabel('[Å]')

        time_ax = ax.text(0.97,0.97, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top")
        time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % 0.00))


        #print(total_frames)
        animation_data = {'t': 0.0, 'ax':ax ,'frame' : 0}
        def func_animation(*arg):
            
            time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (animation_data['t']/femtoseconds)))

            animation_data['t'] = animation_data['t'] + dt
            if animation_data['t'] > self.simulation.total_time:
                animation_data['t'] = 0.0

            #print(animation_data['frame'])
            animation_data['frame'] +=1
            index = int((self.simulation.store_steps)/self.simulation.total_time * animation_data['t'])

            wavefunction_plot.set_data(complex_to_rgba(self.simulation.Ψ_plot[index], max_val= wavefunction_saturation))
            return potential_plot,wavefunction_plot, time_ax

        frame = 0
        a = animation.FuncAnimation(fig, func_animation,
                                    blit=True, frames=total_frames, interval= 1/fps * 1000)
        if save_animation == True:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            a.save('animation.mp4', writer=writer)
        else:
            plt.show()
            
            
        
    def plot3D_SI(self, t, L_norm = 1, Z_norm = 1,unit = 1, contrast_vals= [0.1, 0.25]):

        mlab.figure(bgcolor=(0,0,0), size=(700, 700))
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)   

        psi = self.simulation.Ψ_plot[index]
        psi = np.abs(psi)


        L = self.simulation.H.extent/2/L_norm
        Z = self.simulation.H.z_extent/2/Z_norm
        N = self.simulation.H.N

        #surf = mlab.mesh(self.simulation.H.particle_system.x,self.simulation.H.particle_system.y, psi, colormap='jet')
        surf = mlab.surf(psi,warp_scale="auto",colormap='jet')
        mlab.colorbar(surf, orientation='vertical')

        time_label = mlab.text(0.1, 0.9, '', width=0.5)
        time_label.property.color = (1.0, 1.0, 1.0)
        
        time_label.text = 't = {:.2f} (ms)'.format(t/unit)
        
        #if np.amax(np.abs(self.simulation.Ψ_plot)) == np.amax(np.abs(self.simulation.Ψ_plot[index])):
            #time_label.text = 't-peak = {:.2f} (ms)'.format(t/unit)

        mlab.outline()

        mlab.axes(xlabel='x [mm]', ylabel='y [mm]', zlabel='psi',
                  nb_labels=6, ranges=(-L, L, -L, L, np.min(psi), np.amax(psi)))
        #mlab.view(azimuth=60,elevation=60,distance=N*4)
        colorbar = mlab.colorbar(orientation='vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        file = str(t) + '.png'
        mlab.savefig(file)
        mlab.show()
            
    def plot3D(self, t, L_norm = 1, Z_norm = 1,unit = femtoseconds, contrast_vals= [0.1, 0.25]):

        mlab.figure(bgcolor=(0,0,0), size=(700, 700))
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)   

        psi = self.simulation.Ψ_plot[index]
        psi = np.abs(psi)


        L = self.simulation.H.extent/2/Å/L_norm
        Z = self.simulation.H.z_extent/2/Å/Z_norm
        N = self.simulation.H.N

        #surf = mlab.mesh(self.simulation.H.particle_system.x,self.simulation.H.particle_system.y, psi, colormap='jet')
        surf = mlab.surf(psi,warp_scale="auto",colormap='jet')
        mlab.colorbar(surf, orientation='vertical')

        time_label = mlab.text(0.1, 0.9, '', width=0.5)
        time_label.property.color = (1.0, 1.0, 1.0)
        
        time_label.text = 't = {:.2f} microseconds'.format(t/unit)
        
        if np.max(psi) == 1.00:
            time_label.text = 't-peak = {:.2f} microseconds'.format(t/unit)

        mlab.outline()

        mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',
                  nb_labels=6, ranges=(-L, L, -L, L, np.min(psi), np.max(psi)))
        #mlab.view(azimuth=60,elevation=60,distance=N*4)
        colorbar = mlab.colorbar(orientation='vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        file = str(t) + '.png'
        mlab.savefig(file)
        mlab.show()



    def animate3D(self, L_norm = 1, Z_norm = 1, unit = femtoseconds,time = 'femtoseconds', contrast_vals= [0.1, 0.25]):
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mlab.figure(bgcolor = (0,0,0), size=(700, 700))
        
        if 1==1:
            psi = self.simulation.Ψ[0]
            
            abs_max = self.simulation.Ψmax
            psi = np.abs((psi)/(abs_max))
            
            dt_store = self.simulation.total_time/self.simulation.store_steps


            L = self.simulation.H.extent/2/Å/L_norm
            Z = self.simulation.H.z_extent/2/Å/Z_norm
            N = self.simulation.H.N
            psi = np.where(psi > contrast_vals[1], contrast_vals[1],psi)
            psi = np.where(psi < contrast_vals[0], contrast_vals[0],psi)
            field = mlab.pipeline.scalar_field(psi)
            vol = mlab.pipeline.volume(field)


            # Update the shadow LUT of the volume module.
            vol.update_ctf = True

            mlab.outline()
            x_latex = 'x/w_o'
            y_latex = 'y/w_o'
            z_latex = 'z/\lambda'
            
           # x_latex = mlabtex(0.,0.,.0,x_latex)
            #y_latex = mlabtex(0.,0.,.0,x_latex)
           # z_latex = mlabtex(0.,0.,.0,x_latex)
            surf = mlab.mesh(self.simulation.H.particle_system.x,self.simulation.H.particle_system.y,psi)
            mlab.colorbar(surf,orientation='vertical')
        
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )
            # Define a text label for time
            time_label = mlab.text(0.1, 0.9, '', width=0.3, color=(1, 1, 1))  # White text color
            
            colorbar = mlab.colorbar(orientation='vertical')
            
            def update_time_label(t):
                time_label.text = 'Time: {:.2f} {}'.format(t,time)


            data = {'t': 0.0}
            @mlab.animate(delay=10)
            def animation():
                while (1):
                    data['t'] += 0.05
                    k1 = int(data['t']) % (self.simulation.store_steps)
                    psi = self.simulation.Ψ[k1]

                    psi = np.abs((psi)/(abs_max))

                    
                    surf = mlab.mesh(self.simulation.H.particle_system.x,self.simulation.H.particle_system.y,psi)
                    
                    update_time_label(k1*dt_store/unit)
                    mlab.view(azimuth=60,elevation=60,distance=N*3.5)
                    
                    yield
                    file = str(k1) + '.png'
                    #◘mlab.savefig(file)

            ua = animation()
            mlab.show()
     
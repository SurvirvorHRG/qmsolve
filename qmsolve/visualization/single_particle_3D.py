import numpy as np
from mayavi import mlab
from .visualization import Visualization
from ..util.colour_functions import complex_to_rgb, complex_to_rgba
from ..util.constants import *
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import animation
from mlabtex import mlabtex



class VisualizationSingleParticle3D(Visualization):

    def __init__(self,eigenstates):
        self.eigenstates = eigenstates
        self.plot_type = 'volume'

    def slider_plot(self):
        raise NotImplementedError

    def plot_eigenstate(self, k, contrast_vals= [0.1, 0.25]):
        eigenstates = self.eigenstates.array
        mlab.figure(1, bgcolor=(0, 0, 0), size=(700, 700))
        psi = eigenstates[k]

        if self.plot_type == 'volume':
            
            abs_max= np.amax(np.abs(eigenstates))
            psi = (psi)/(abs_max)

            L = self.eigenstates.extent/2/Å
            N = self.eigenstates.N

            vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(psi))

            # Change the color transfer function
            from tvtk.util import ctf
            c = ctf.save_ctfs(vol._volume_property)
            c['rgb'] = [[-0.45, 0.3, 0.3, 1.0],
                        [-0.4, 0.1, 0.1, 1.0],
                        [-0.3, 0.0, 0.0, 1.0],
                        [-0.2, 0.0, 0.0, 1.0],
                        [-0.001, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.001, 1.0, 0.0, 0.],
                        [0.2, 1.0, 0.0, 0.0],
                        [0.3, 1.0, 0.0, 0.0],
                        [0.4, 1.0, 0.1, 0.1],
                        [0.45, 1.0, 0.3, 0.3]]

            c['alpha'] = [[-0.5, 1.0],
                          [-contrast_vals[1], 1.0],
                          [-contrast_vals[0], 0.0],
                          [0, 0.0],
                          [contrast_vals[0], 0.0],
                          [contrast_vals[1], 1.0],
                         [0.5, 1.0]]
            ctf.load_ctfs(c, vol._volume_property)
            # Update the shadow LUT of the volume module.
            vol.update_ctf = True

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )
            #azimuth angle
            φ = 30
            mlab.view(azimuth= φ,  distance=N*3.5)
            mlab.show()


        if self.plot_type == 'abs-volume':
            
            abs_max= np.amax(np.abs(eigenstates))
            psi = (psi)/(abs_max)

            L = self.eigenstates.extent/2/Å
            N = self.eigenstates.N

            vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.abs(psi)), vmin= contrast_vals[0], vmax= contrast_vals[1])
            # Change the color transfer function

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )
            #azimuth angle
            φ = 30
            mlab.view(azimuth= φ,  distance=N*3.5)
            mlab.show()




        elif self.plot_type == 'contour':
            psi = eigenstates[k]
            L = self.eigenstates.extent/2/Å
            N = self.eigenstates.N
            isovalue = np.mean(contrast_vals)
            abs_max= np.amax(np.abs(eigenstates))
            psi = (psi)/(abs_max)

            field = mlab.pipeline.scalar_field(np.abs(psi))

            arr = mlab.screenshot(antialiased = False)

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )
            colour_data = np.angle(psi.T.ravel())%(2*np.pi)
            field.image_data.point_data.add_array(colour_data)
            field.image_data.point_data.get_array(1).name = 'phase'
            field.update()
            field2 = mlab.pipeline.set_active_attribute(field, 
                                                        point_scalars='scalar')
            contour = mlab.pipeline.contour(field2)
            contour.filter.contours= [isovalue,]
            contour2 = mlab.pipeline.set_active_attribute(contour, 
                                                        point_scalars='phase')
            s = mlab.pipeline.surface(contour, colormap='hsv', vmin= 0.0 ,vmax= 2.*np.pi)

            s.scene.light_manager.light_mode = 'vtk'
            s.actor.property.interpolation = 'phong'


            #azimuth angle
            φ = 30
            mlab.view(azimuth= φ,  distance=N*3.5)

            mlab.show()

    def animate(self,  contrast_vals= [0.1, 0.25]):
        eigenstates = self.eigenstates.array
        energies = self.eigenstates.energies
        mlab.figure(1, bgcolor=(0, 0, 0), size=(700, 700))

        
        if self.plot_type == 'volume':
            psi = eigenstates[0]
            
            abs_max= np.amax(np.abs(eigenstates))
            psi = (psi)/(abs_max)


            L = self.eigenstates.extent/2/Å
            N = self.eigenstates.N
            field = mlab.pipeline.scalar_field(psi)
            vol = mlab.pipeline.volume(field)

            color1 = complex_to_rgb(np.exp( 1j*2*np.pi/10*0)) 
            color2 = complex_to_rgb(-np.exp( 1j*2*np.pi/10*0)) 

            # Change the color transfer function
            from tvtk.util import ctf
            c = ctf.save_ctfs(vol._volume_property)
            c['rgb'] = [[-0.45, *color1],
                        [-0.4, *color1],
                        [-0.3, *color1],
                        [-0.2, *color1],
                        [-0.001, *color1],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.001, *color2],
                        [0.2, *color2],
                        [0.3, *color2],
                        [0.4, *color2],
                        [0.45, *color2]]

            c['alpha'] = [[-0.5, 1.0],
                          [-contrast_vals[1], 1.0],
                          [-contrast_vals[0], 0.0],
                          [0, 0.0],
                          [contrast_vals[0], 0.0],
                          [contrast_vals[1], 1.0],
                         [0.5, 1.0]]
            ctf.load_ctfs(c, vol._volume_property)
            # Update the shadow LUT of the volume module.
            vol.update_ctf = True

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )

            #azimuth angle
            φ = 30
            mlab.view(azimuth= φ,  distance=N*3.5)


            data = {'t': 0.0}
            @mlab.animate(delay=10)
            def animation():
                while (1):
                    data['t'] += 0.05
                    k1 = int(data['t']) % len(energies)
                    k2 = (int(data['t']) + 1) % len(energies)
                    if data['t'] % 1.0 > 0.5:
                        t = (data['t'] - int(data['t']) - 0.5)
                        psi = (np.cos(np.pi*t)*eigenstates[k1]
                            + np.sin(np.pi*t)*eigenstates[k2])

                        color1 = complex_to_rgb(np.exp( 1j*2*np.pi/10*k1)*np.cos(np.pi*t) + np.exp( 1j*2*np.pi/10*k2)*np.sin(np.pi*t)) 
                        color2 = complex_to_rgb(-np.exp( 1j*2*np.pi/10*k1)*np.cos(np.pi*t) - np.exp( 1j*2*np.pi/10*k2)*np.sin(np.pi*t)) 
                    else:
                        psi = eigenstates[k1]
                        color1 = complex_to_rgb(np.exp( 1j*2*np.pi/10*k1)) 
                        color2 = complex_to_rgb(-np.exp( 1j*2*np.pi/10*k1)) 

                    psi = (psi)/(abs_max)
                    field.mlab_source.scalars = psi
                    # Change the color transfer function
                    from tvtk.util import ctf
                    c = ctf.save_ctfs(vol._volume_property)
                    c['rgb'] = [[-0.45, *color1],
                                [-0.4, *color1],
                                [-0.3, *color1],
                                [-0.2, *color1],
                                [-0.001, *color1],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.001, *color2],
                                [0.2, *color2],
                                [0.3, *color2],
                                [0.4, *color2],
                                [0.45, *color2]]

                    c['alpha'] = [[-0.5, 1.0],
                                  [-contrast_vals[1], 1.0],
                                  [-contrast_vals[0], 0.0],
                                  [0, 0.0],
                                  [contrast_vals[0], 0.0],
                                  [contrast_vals[1], 1.0],
                                 [0.5, 1.0]]
                    ctf.load_ctfs(c, vol._volume_property)
                    # Update the shadow LUT of the volume module.
                    vol.update_ctf = True

                    φ = 30 + data['t'] * 360 / 10 
                    mlab.view(azimuth= φ, distance=N*3.5)

                    yield

            ua = animation()
            mlab.show()


        if self.plot_type == 'abs-volume':
            psi = eigenstates[0]
            
            abs_max= np.amax(np.abs(eigenstates))
            psi = np.abs((psi)/(abs_max))


            L = self.eigenstates.extent/2/Å
            N = self.eigenstates.N
            psi = np.where(psi > contrast_vals[1], contrast_vals[1],psi)
            psi = np.where(psi < contrast_vals[0], contrast_vals[0],psi)
            field = mlab.pipeline.scalar_field(psi)
            vol = mlab.pipeline.volume(field)


            # Update the shadow LUT of the volume module.
            vol.update_ctf = True

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )

            #azimuth angle
            φ = 30
            mlab.view(azimuth= φ,  distance=N*3.5)


            data = {'t': 0.0}
            @mlab.animate(delay=10)
            def animation():
                while (1):
                    data['t'] += 0.05
                    k1 = int(data['t']) % len(energies)
                    k2 = (int(data['t']) + 1) % len(energies)
                    if data['t'] % 1.0 > 0.5:
                        t = (data['t'] - int(data['t']) - 0.5)
                        psi = (np.cos(np.pi*t)*eigenstates[k1]
                            + np.sin(np.pi*t)*eigenstates[k2])
                    else:
                        psi = eigenstates[k1]

                    psi = np.abs((psi)/(abs_max))
                    psi = np.where(psi > contrast_vals[1], contrast_vals[1],psi)
                    psi = np.where(psi < contrast_vals[0], contrast_vals[0],psi)

                    field.mlab_source.scalars = psi
                    # Change the color transfer function

                    φ = 30 + data['t'] * 360 / 10 
                    mlab.view(azimuth= φ, distance=N*3.5)

                    yield

            ua = animation()
            mlab.show()



        elif self.plot_type == 'contour':
            psi = eigenstates[0]
            L = self.eigenstates.extent/2/Å
            N = self.eigenstates.N
            isovalue = np.mean(contrast_vals)


            abs_max= np.amax(np.abs(eigenstates))
            psi = (psi)/(abs_max)

            field = mlab.pipeline.scalar_field(np.abs(psi))

            arr = mlab.screenshot(antialiased = False)

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )
            colour_data = np.angle(psi.T.ravel())%(2*np.pi)
            field.image_data.point_data.add_array(colour_data)
            field.image_data.point_data.get_array(1).name = 'phase'
            field.update()
            field2 = mlab.pipeline.set_active_attribute(field, 
                                                        point_scalars='scalar')
            contour = mlab.pipeline.contour(field2)
            contour.filter.contours= [isovalue,]
            contour2 = mlab.pipeline.set_active_attribute(contour, 
                                                        point_scalars='phase')
            s = mlab.pipeline.surface(contour2, colormap='hsv', vmin= 0.0 ,vmax= 2.*np.pi)

            s.scene.light_manager.light_mode = 'vtk'
            s.actor.property.interpolation = 'phong'


            #azimuth angle
            φ = 30
            mlab.view(azimuth= φ,  distance=N*3.5)




            data = {'t': 0.0}
            @mlab.animate(delay=10)
            def animation():
                while (1):
                    data['t'] += 0.05
                    k1 = int(data['t']) % len(energies)
                    k2 = (int(data['t']) + 1) % len(energies)
                    if data['t'] % 1.0 > 0.5:
                        t = (data['t'] - int(data['t']) - 0.5)
                        psi = (np.cos(np.pi*t)*eigenstates[k1]*np.exp( 1j*2*np.pi/10*k1) 
                             + np.sin(np.pi*t)*eigenstates[k2]*np.exp( 1j*2*np.pi/10*k2))


                    else:
                        psi = eigenstates[k1]*np.exp( 1j*2*np.pi/10*k1)
                    psi = (psi)/(abs_max)
                    np.copyto(colour_data, np.angle(psi.T.ravel())%(2*np.pi))
                    field.mlab_source.scalars = np.abs(psi)

                    φ = 30 + data['t'] * 360 / 10 
                    mlab.view(azimuth= φ, distance=N*3.5)


                    yield
            ua = animation()
            mlab.show()







    def superpositions(self, states, contrast_vals= [0.1, 0.25], **kw):

        params = {'dt': 0.1}
        for k in kw.keys():
            if k in params:
                params[k] = kw[k]
            else:
                raise KeyError

        
        coeffs = states
        eigenstates = self.eigenstates.array
        energies = self.eigenstates.energies
        mlab.figure(1, bgcolor=(0, 0, 0), size=(700, 700))
        psi = sum([eigenstates[i]*coeffs[i] for i in range(len(coeffs))])

        if self.plot_type == 'volume':
            raise NotImplementedError
        elif self.plot_type == 'abs-volume':
            abs_max= np.amax(np.abs(eigenstates))
            psi = np.abs((psi)/(abs_max))


            L = self.eigenstates.extent/2/Å
            N = self.eigenstates.N
            psi = np.where(psi > contrast_vals[1], contrast_vals[1],psi)
            psi = np.where(psi < contrast_vals[0], contrast_vals[0],psi)
            field = mlab.pipeline.scalar_field(psi)
            vol = mlab.pipeline.volume(field)


            # Update the shadow LUT of the volume module.
            vol.update_ctf = True

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )

            #azimuth angle
            φ = 30
            mlab.view(azimuth= φ,  distance=N*3.5)
            data = {'t': 0.0}

            @mlab.animate(delay=10)
            def animation():
                while (1):
                    data['t'] += params['dt']
                    t = data['t']
                    psi = sum([eigenstates[i]*np.exp(-1.0j*energies[i]*t)*coeffs[i]
                            for i in range(len(coeffs))])
                    psi = np.abs((psi)/(abs_max))

                    psi = np.where(psi > contrast_vals[1], contrast_vals[1],psi)
                    psi = np.where(psi < contrast_vals[0], contrast_vals[0],psi)
                    field.mlab_source.scalars = psi

                    φ = 30 + data['t'] * 360 / 10 
                    mlab.view(azimuth= φ, distance=N*3.5)

                    yield

            animation()
            mlab.show()
        elif self.plot_type == 'contour':
            L = self.eigenstates.extent/2/Å
            N = self.eigenstates.N
            isovalue = np.mean(contrast_vals)


            abs_max= np.amax(np.abs(eigenstates))
            psi = (psi)/(abs_max)

            field = mlab.pipeline.scalar_field(np.abs(psi))

            arr = mlab.screenshot(antialiased = False)

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )
            colour_data = np.angle(psi.T.ravel())%(2*np.pi)
            field.image_data.point_data.add_array(colour_data)
            field.image_data.point_data.get_array(1).name = 'phase'
            field.update()
            field2 = mlab.pipeline.set_active_attribute(field, 
                                                        point_scalars='scalar')
            contour = mlab.pipeline.contour(field2)
            contour.filter.contours= [isovalue,]
            contour2 = mlab.pipeline.set_active_attribute(contour, 
                                                        point_scalars='phase')
            s = mlab.pipeline.surface(contour2, colormap='hsv', vmin= 0.0 ,vmax= 2.*np.pi)

            s.scene.light_manager.light_mode = 'vtk'
            s.actor.property.interpolation = 'phong'


            #azimuth angle
            φ = 30
            mlab.view(azimuth= φ,  distance=N*3.5)
            data = {'t': 0.0}

            @mlab.animate(delay=10)
            def animation():
                while (1):
                    data['t'] += params['dt']
                    t = data['t']
                    psi = sum([eigenstates[i]*np.exp(-1.0j*energies[i]*t)*coeffs[i]
                               for i in range(len(coeffs))])

                    psi = (psi)/(abs_max)
                    np.copyto(colour_data, np.angle(psi.T.ravel())%(2*np.pi))
                    field.mlab_source.scalars = np.abs(psi)

                    φ = 30 + data['t'] * 360 / 10 
                    mlab.view(azimuth= φ, distance=N*3.5)
                    yield
            animation()
            mlab.show()
            

from .visualization import TimeVisualization

class TimeVisualizationSingleParticle3D(TimeVisualization):
    def __init__(self,simulation):
        self.simulation = simulation
        self.H = simulation.H
        self.plot_type = 'abs-volume'

    def subplot2D_hot(self, t, L_norm = 1, Z_norm = 1,unit = milliseconds, figsize=(15, 15)):


        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        H = self.simulation.H
        x, y, z  = np.mgrid[ -H.extent/2: H.extent/2:H.N*1j, -H.extent/2: H.extent/2:H.N*1j, -H.z_extent/2: H.z_extent/2:H.Nz*1j]
        X = self.simulation.H.particle_system.x
        Y = self.simulation.H.particle_system.y
        Z = self.simulation.H.particle_system.z
        
        plt.style.use("classic")
        
        # Number of subplots
        N = 10  # You can change this value to 100 or 150 as needed
        
        # Select N evenly spaced z values
        z_indices = np.linspace(0, Z.shape[2] - 1, N).astype(int)
        selected_z_values = z[z_indices]
        
        # Create subplots
        fig, axes = plt.subplots(1, N, figsize=(N * 5, 5), sharex=True, sharey=True)
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)
        toplot=abs(self.simulation.Ψ_plot[index]) 
        
        for i, z_val in enumerate(selected_z_values):
            ax = axes[i]
            contour = ax.contourf(X[:, :, z_indices[i]]/L_norm, Y[:, :, z_indices[i]]/L_norm, toplot[:, :, z_indices[i]], levels=50, cmap='jet')
            z_norm = Z[0,0,z_indices[i]]/Z_norm
            #print(z_val)
            #ax.set_title(f'z = {z_val:.2f}')
            ax.set_title('$t= %.2f$ (ms) at $z/\lambda= %.2f$' % (t/unit, z_norm))
            ax.set_xlabel('$x/w_o$')
            if i == 0:
                ax.set_ylabel('$y/w_o$')
            
            # Add a colorbar for each subplot
            cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
            cbar.set_label('$|\psi|$')
        
        plt.tight_layout()
        #plt.show()
        """
        #Two-dimensional countour map
        fig = plt.figure("Contour map")    # figure
        plt.clf()                       # clears the figure
        fig.set_size_inches(8,6)
        plt.axes().set_aspect('equal')  # Axes aspect ratio
        
        plt.xlabel("$x/w_o$",fontsize = 14)               # choose axes labels
        plt.ylabel("$z/\lambda$",fontsize = 14)
    
        # Makes the contour plot:
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)
        mid = int(self.simulation.H.N / 2) - 1
        toplot=abs(self.simulation.Ψ_plot[index][:,mid,:])      
        from matplotlib import cm
        plt.contourf(self.simulation.H.particle_system.x[:,0,0]/L_norm, self.simulation.H.particle_system.z[0,0,:]/Z_norm, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        cbar=plt.colorbar()          # colorbar
    
        
        cbar.set_label('$|\psi|$',fontsize=14)
        plt.title('$t= %.2f$ (ps) at $y/w_o= %.2f$' % (t * 1e12, self.simulation.H.particle_system.y[0, mid, 0]/L_norm))    # Title of the plot
        """
        plt.show()
        
    
        
    def final_plot3D_hot(self,L_norm = 1, Z_norm = 1,unit = milliseconds, figsize=(15, 15),time="ms"):
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        
        z = self.simulation.H.particle_system.z[0,0,:]
        total_time = self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt
        tvec=np.linspace(0,self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt,self.simulation.store_steps+1)
        tt,zz=np.meshgrid(tvec,z)
        
        # Generate the 3D plot
        fig = plt.figure("Evolution of 1D cut at y=0")
        ax = fig.add_subplot(111, projection='3d')
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        #toplot= np.abs(self.simulation.Ψ_plot[:,49,49,:])**2
        mid = int(self.simulation.H.N / 2) - 1
        toplot= np.abs(self.simulation.Ψ_plot[:,mid,mid,:])
        toplot = toplot.T

        # Generates the plot
        # Plot the surface
        surf = ax.plot_surface(zz/Z_norm,tt/unit,toplot, cmap=cm.jet, linewidth=0, antialiased=False)
        
        # Add colorbar and labels
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.set_label('$|\psi|$', fontsize=14)
        ax.set_xlabel('$z/\lambda$')
        ax.set_ylabel('$t\ (ms)$')
        ax.set_zlabel('$|Ψ|^2$')


        
        L = self.simulation.H.extent/2/L_norm
        Z = self.simulation.H.z_extent/2/Z_norm
       
        plt.show()      # Displays figure on screen
        
        mlab.figure(fgcolor=(0,0,0),bgcolor=(0.9,0.9,0.9),size=(700, 700))
        L = self.simulation.H.extent/2/L_norm
        Z = self.simulation.H.z_extent/2/Z_norm
        N = self.simulation.H.N
        
        surf = mlab.surf(zz/Z_norm,tt/unit,toplot,warp_scale="auto",colormap='jet')
        mlab.colorbar(surf,orientation='vertical')  
              
       #♠ mlab.outline()
        
         
        x_latex = '$z/\lambda$'
        y_latex = '$T\ (ms)$'
        z_latex = ''
         
        ax = mlab.axes(xlabel=x_latex, ylabel=y_latex,zlabel = z_latex,nb_labels=6 , ranges = (-Z,Z,0,total_time/unit,0,1))
        #ax.label_text_property.font_size = 50
        #ax.label_text_property.font_size = 20
        #ax.title_text_property.font_size = 20
        ax.axes.font_factor = 1.9
        ax.label_text_property.font_family = 'times'
        ax.title_text_property.font_family = 'times'
        ax.axes.y_axis_visibility = False
        ax.axes.label_format = '%-#6.2g'
        
        

        
        colorbar = mlab.colorbar(nb_labels=6,orientation = 'vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        colorbar_label = colorbar.scalar_bar.label_text_property
        colorbar_label.font_family = 'times'
        colorbar.scalar_bar.unconstrained_font_size = True
        colorbar.scalar_bar.label_format = '%.2f'
        colorbar_label.font_size = 22
        mlab.show()
        

    def final_plot_hot(self,L_norm = 1, Z_norm = 1,unit = milliseconds, figsize=(15, 15)):
        
        from mpl_toolkits.mplot3d import Axes3D
        
        z = self.simulation.H.particle_system.z[0,0,:]
        tvec=np.linspace(0,self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt,self.simulation.store_steps+1)
        tt,zz=np.meshgrid(tvec,z)
        plt.figure("Evolution of 1D cut at y=0")              # figure
        plt.clf()                       # clears the figure
        
        # Generates the plot
        
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mid = int(self.simulation.H.N / 2) - 1
        toplot= np.abs(self.simulation.Ψ_plot[:,mid,mid,:])
        toplot = toplot.T
        from matplotlib import cm
        plt.contourf(zz/Z_norm,tt/unit, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        L = self.simulation.H.extent/2/Å/L_norm
        Z = self.simulation.H.z_extent/2/Å/Z_norm
        
        cbar=plt.colorbar()               # colorbar
        
        
        plt.xlabel('$z/\lambda$')               # choose axes labels, title of the plot and axes range
        plt.ylabel('$t\ (ms)$')
        cbar.set_label('$|\psi|^2$',fontsize=14)
        
        plt.show()      # Displays figure on screen
        
    def final_plot3D(self,L_norm = 1, Z_norm = 1,unit = milliseconds, figsize=(15, 15),time="ns"):
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        
        z = self.simulation.H.particle_system.z[0,0,:]
        total_time = self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt
        tvec=np.linspace(0,self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt,self.simulation.store_steps+1)
        tt,zz=np.meshgrid(tvec,z)
        
        # Generate the 3D plot
        fig = plt.figure("Evolution of 1D cut at y=0")
        ax = fig.add_subplot(111, projection='3d')
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mid = int(self.simulation.H.N / 2) + 1
        toplot= np.abs(self.simulation.Ψ_plot[:,mid,mid,:])
        toplot = toplot.T

        # Generates the plot
        # Plot the surface
        surf = ax.plot_surface(zz/Z_norm/Å, tt/unit, toplot, cmap=cm.jet, linewidth=0, antialiased=False)
        
        # Add colorbar and labels
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.set_label('$|\psi|^2$', fontsize=14)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$t\ (ns)$')
        ax.set_zlabel('$|Ψ|^2$')


        
        L = self.simulation.H.extent/2/Å/L_norm
        Z = self.simulation.H.z_extent/2/Å/Z_norm
        #tmp = ax.imshow(toplot, cmap='jet')  
        #fig.colorbar(tmp,ax = ax)
       
        plt.show()      # Displays figure on screen
        
        mlab.figure(bgcolor=(0,0,0), size=(1400, 1400))
        L = self.simulation.H.extent/2/Å/L_norm
        Z = self.simulation.H.z_extent/2/Å/Z_norm
        N = self.simulation.H.N
        
        #surf = mlab.mesh(zz/Z_norm/Å,tt/unit,toplot,colormap='jet')
        surf = mlab.surf(toplot,warp_scale="auto",colormap='jet')
        #surf = mlab.surf(toplot,warp_scale="auto",colormap='jet')
        
        #surf = mlab.surf(psi[:,:,49],colormap='jet')
        mlab.colorbar(surf,title='psi',orientation='vertical')  
              
        mlab.outline()
        
        x_latex = 'z'
        y_latex = 't (ms)'
        z_latex = ''
         
        mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=3 , ranges = (-Z,Z,0,total_time/unit,np.min(toplot),np.max(toplot)) )
        #mlab.axes(xlabel='x [Å]', ylabel='y [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
        colorbar = mlab.colorbar(orientation = 'vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        #file = str(t) + '.png'
        #mlab.savefig(file)
        mlab.show()
        

    def final_plot(self,L_norm = 1, Z_norm = 1,unit = milliseconds, figsize=(15, 15),time="ms",fixmaximum = 0.0):
        
        from mpl_toolkits.mplot3d import Axes3D
        
        z = self.simulation.H.particle_system.z[0,0,:]
        tvec=np.linspace(0,self.simulation.Nt_per_store_step*self.simulation.store_steps*self.simulation.dt,self.simulation.store_steps+1)
        tt,zz=np.meshgrid(tvec,z)
        plt.figure("Evolution of 1D cut at y=0")              # figure
        plt.clf()                       # clears the figure
        
        # Generates the plot
        
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mid = int(self.simulation.H.N / 2) - 1
        toplot= np.abs(self.simulation.Ψ_plot[:,mid,mid,:])
        toplot = toplot.T
        if fixmaximum>0:
            toplot[toplot>fixmaximum]=fixmaximum 
        
        from matplotlib import cm
        plt.contourf(zz/Z_norm, tt/unit, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)

        cbar=plt.colorbar()               # colorbar
        
        plt.xlabel('$z$')               # choose axes labels, title of the plot and axes range
        plt.ylabel('$t\ (ms)$')
        cbar.set_label('$|\psi|^2$',fontsize=14)
        plt.show()      # Displays figure on screen

    def plot2D_hot(self, t, L_norm = 1, Z_norm = 1,unit = femtoseconds, figsize=(15, 15)):


        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("classic")
        
        #Two-dimensional countour map
        fig = plt.figure("Contour map")    # figure
        plt.clf()                       # clears the figure
        fig.set_size_inches(8,6)
        plt.axes().set_aspect('equal')  # Axes aspect ratio
        
        plt.xlabel("$x/w_o$",fontsize = 14)               # choose axes labels
        plt.ylabel("$z/\lambda$",fontsize = 14)
    
        # Makes the contour plot:
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)
        mid = int(self.simulation.H.N / 2) - 1
        toplot=abs(self.simulation.Ψ_plot[index][:,mid,:])      
        from matplotlib import cm
        plt.contourf(self.simulation.H.particle_system.x[:,0,0]/L_norm, self.simulation.H.particle_system.z[0,0,:]/Z_norm, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        cbar=plt.colorbar()          # colorbar
    
        
        cbar.set_label('$|\psi|$',fontsize=14)
        plt.title('$t= %.2f$ (ps) at $y/w_o= %.2f$' % (t * 1e12, self.simulation.H.particle_system.y[0, mid, 0]/L_norm))    # Title of the plot
        plt.show()

    def plot2D_xz(self, t, L_norm = 1, Z_norm = 1,unit = femtoseconds, figsize=(15, 15)):


        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("classic")
        
        #Two-dimensional countour map
        fig = plt.figure("Contour map")    # figure
        plt.clf()                       # clears the figure
        fig.set_size_inches(8,6)
        plt.axes().set_aspect('equal')  # Axes aspect ratio
        
        plt.xlabel("$x\ (mm)$",fontsize = 14)               # choose axes labels
        plt.ylabel("$z\ (mm)$",fontsize = 14)
    
        # Makes the contour plot:
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)
        mid = int(self.simulation.H.N / 2) - 1
        toplot=abs(self.simulation.Ψ_plot[index][:,mid,:])      
        from matplotlib import cm
        plt.contourf(self.simulation.H.particle_system.x[:,0,0]/L_norm/Å/1e7, self.simulation.H.particle_system.z[0,0,:]/Z_norm/Å/1e7, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        cbar=plt.colorbar()          # colorbar
    
        
        cbar.set_label('$|\psi|$',fontsize=14)
        plt.title('$t= %.2f$ at $y= %.2f\ (mm)$' % (t/unit, self.simulation.H.particle_system.y[0, mid, 0]/L_norm/Å/1e7))    # Title of the plot
        plt.show()
        
    def plot2D_xy(self, t, L_norm = 1, Z_norm = 1,unit = milliseconds, figsize=(15, 15)):


        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("classic")
        
        #Two-dimensional countour map
        fig = plt.figure("Contour map")    # figure
        plt.clf()                       # clears the figure
        fig.set_size_inches(8,6)
        plt.axes().set_aspect('equal')  # Axes aspect ratio
        
        plt.xlabel("$x\ (mm)$",fontsize = 14)               # choose axes labels
        plt.ylabel("$y\ (mm)$",fontsize = 14)
    
        # Makes the contour plot:
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)
        mid = int(self.simulation.H.Nz / 2) - 1
        toplot=abs(self.simulation.Ψ_plot[index][:,:,mid])      
        from matplotlib import cm
        plt.contourf(self.simulation.H.particle_system.x[:,0,0]/L_norm/Å/1e7, self.simulation.H.particle_system.y[0,:,0]/L_norm/Å/1e7, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
        cbar=plt.colorbar()          # colorbar
    
        
        cbar.set_label('$|\psi|$',fontsize=14)
        plt.title('$t= %.2f$ (ms) at $z= %.2f\ (mm)$' % (t/unit, self.simulation.H.particle_system.z[0, 0, mid]/Z_norm/Å/1e7))    # Title of the plot
        plt.show()

        

    def plot3D(self, t, L_norm = 1, Z_norm = 1,unit = milliseconds):
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)   
        psi = self.simulation.Ψ_plot[index]
        mid = int(self.simulation.H.Nz / 2) - 1
        psi = np.abs(psi[:,:,mid])

         

        #surf = mlab.mesh(self.simulation.H.particle_system.x[:,:,0],self.simulation.H.particle_system.y[:,:,0],psi[:,:,49],colormap='jet') 
        mlab.figure(bgcolor=(0,0,0), size=(900, 900))
        L = self.simulation.H.extent/2/Å/L_norm/1e7
        N = self.simulation.H.N
        
        #surf = mlab.mesh(zz/Z_norm/Å,tt/unit,toplot,colormap='jet')
        warp_coef = 100 + np.amax(psi) * 1e2
        #warp_coef = 50
        surf = mlab.surf(psi,warp_scale=warp_coef,colormap='jet')
        #surf = mlab.surf(toplot,warp_scale="auto",colormap='jet')
        
        #surf = mlab.surf(psi[:,:,49],colormap='jet')
        #mlab.colorbar(surf,title='psi',orientation='vertical')  
        #time_label.text = 'Time: {:.2f} {}'.format(t/unit)
        time_label = mlab.text(0.1,0.9,'',width=0.2)
        time_label.property.color = (1.0,1.0,1.0)
        time_label.text = '$t = {:.2f}\ (ms)$'.format(t/unit)
        #print(np.amax(psi))
        #print(self.simulation.Ψmax)
        #print(np.amax(np.abs(self.simulation.Ψ_plot)))
        #print(np.amax(np.abs(self.simulation.Ψ_plot[index])))
        if np.amax(np.abs(self.simulation.Ψ_plot)) == np.amax(np.abs(self.simulation.Ψ_plot[index])):
            time_label.text = '$t-peak = {:.2f}\ (ms)$'.format(t/unit)
              
        #mlab.outline()
        
        time_label = mlab.text(0.1, 0.9, '', width=0.3, color=(1, 1, 1))  # White text color
        
        #mlab.xlabel('x')
        #mlab.ylabel('t')
        #mlab.zlabel('|\psi|^2')
         
        x_latex = '$x\ (mm)$'
        y_latex = '$y\ (mm)$'
        z_latex = ''
         
        ax = mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=3 , ranges = (-L,L,-L,L,np.min(psi),np.amax(psi)) )
        ax.axes.y_axis_visibility = False
        colorbar = mlab.colorbar(orientation = 'vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        file = str(t) + '.png'
        mlab.savefig(file)
        mlab.show()
        
        
        

      
    def animate3D(self, L_norm = 1, Z_norm = 1, unit = femtoseconds,time = 'femtoseconds', contrast_vals= [0.1, 0.25]):
        #self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mlab.figure(1,bgcolor = (0,0,0), size=(700, 700))
        
        if self.plot_type == 'abs-volume':
            psi = self.simulation.Ψ[0]
            
            abs_max = self.simulation.Ψmax
            psi = np.abs((psi)/(abs_max))
            
            dt_store = self.simulation.total_time/self.simulation.store_steps


            L = self.simulation.H.extent/2/Å/L_norm
            Z = self.simulation.H.z_extent/2/Å/Z_norm
            N = self.simulation.H.N
            
            surf = mlab.surf(psi[:,:,49],warp_scale="auto",colormap='jet')
            mlab.colorbar(surf,orientation='vertical')  
            
            mlab.outline()


            x_latex = 'x/w_o'
            y_latex = 'y/w_o'
            z_latex = 'z/\lambda'
            
           # x_latex = mlabtex(0.,0.,.0,x_latex)
            #y_latex = mlabtex(0.,0.,.0,x_latex)
           # z_latex = mlabtex(0.,0.,.0,x_latex)
            
        
            mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
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
                    
                    surf = mlab.surf(psi[:,:,49],warp_scale="auto",colormap='jet')
                    #mlab.colorbar(surf,orientation='vertical')  
                     
                    

                    
                    
    
                    
                    update_time_label(k1*dt_store/unit)
                    mlab.view(azimuth=60,elevation=60,distance=N*3.5)
                    
                    yield
                    file = str(k1) + '.png'
                    #◘mlab.savefig(file)

            ua = animation()
            mlab.show()
        
        

    def plot_hot(self, t, L_norm = 1, Z_norm = 1,figsize=(7, 7),unit = milliseconds, contrast_vals= [0.1, 0.25]):

        mlab.figure(fgcolor=(0.1,0.1,0.1),bgcolor=(0.9,0.9,0.9),size=(700, 700))
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)   
        """
        psi = self.simulation.Ψ_plot[index]
        L = self.simulation.H.extent/2/Å/L_norm
        Z = self.simulation.H.z_extent/2/Å/Z_norm

        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.abs(psi)), vmin= contrast_vals[0], vmax= contrast_vals[1])

        mlab.outline()
        mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
        mlab.show()
        """
        if self.plot_type == 'abs-volume':
            
            psi = self.simulation.Ψ_plot[index]
            
            
           # print(psi)

            L = self.simulation.H.extent/2/L_norm
            Z = self.simulation.H.z_extent/2/Z_norm
            N = self.simulation.H.N

            vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.abs(psi)), vmin= contrast_vals[0], vmax= contrast_vals[1])          
            
            time_label = mlab.text(0.1,0.9,'',width=0.17)
            time_label.property.color = (0.1,0.1,0.1)
            time_label.text = '$T = {:.2f}\ ms$'.format(t/unit)
            time_label.property.font_family = 'times'

            mlab.outline()
            

            
            x_latex = '$x/w_o$'
            y_latex = '$y/w_o$'
            z_latex = '$z/\lambda$'
            

        
            ax =mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
            ax.axes.font_factor = 1.9
            ax.label_text_property.font_family = 'times'
            ax.title_text_property.font_family = 'times'
            ax.axes.y_axis_visibility = False
            ax.axes.label_format = '%-#6.2g'
            
            colorbar = mlab.colorbar(nb_labels=6,orientation = 'vertical')
            colorbar.scalar_bar_representation.position = [0.85, 0.1]
            colorbar_label = colorbar.scalar_bar.label_text_property
            colorbar_label.font_family = 'times'
            colorbar.scalar_bar.label_format = '%.2f'
            colorbar.scalar_bar.unconstrained_font_size = True
            colorbar_label.font_size = 21
            #▒colorbar.scalar_bar.height = 20
            mlab.view(azimuth=60,elevation=60,distance=N*4)
            file = str(t/unit) + '.png'
            mlab.savefig(file)
            mlab.show()
            
            
    def plot(self, t, L_norm = 1, Z_norm = 1,figsize=(7, 7),unit = femtoseconds, contrast_vals= [0.1, 0.25]):

        mlab.figure(1,bgcolor=(0,0,0), size=(700, 700))
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        index = int((self.simulation.store_steps)/self.simulation.total_time*t)   
        """
        psi = self.simulation.Ψ_plot[index]
        L = self.simulation.H.extent/2/Å/L_norm
        Z = self.simulation.H.z_extent/2/Å/Z_norm

        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.abs(psi)), vmin= contrast_vals[0], vmax= contrast_vals[1])

        mlab.outline()
        mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
        mlab.show()
        """
        if self.plot_type == 'abs-volume':
            
            psi = self.simulation.Ψ_plot[index]
            
            
           # print(psi)

            L = self.simulation.H.extent/2/Å/L_norm
            Z = self.simulation.H.z_extent/2/Å/Z_norm
            N = self.simulation.H.N

            vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.abs(psi)), vmin= contrast_vals[0], vmax= contrast_vals[1])          
            
            time_label = mlab.text(0.1,0.9,'',width=0.2)
            time_label.property.color = (1.0,1.0,1.0)
            time_label.text = 'Time: {:.2f} ns'.format(t/unit)

            mlab.outline()
            
            x_latex = 'x/w_o'
            y_latex = 'y/w_o'
            z_latex = 'z/lambda'
            
            x_latex = '$x/w_o$'
            y_latex = '$y/w_o$'
            z_latex = '$z/\lambda$'
            
           # x_latex = mlabtex(0.,0.,.0,x_latex)
            #y_latex = mlabtex(0.,0.,.0,x_latex)
           # z_latex = mlabtex(0.,0.,.0,x_latex)
            
        
            #mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
            mlab.view(azimuth=60,elevation=60,distance=N*4)
            colorbar = mlab.colorbar(orientation = 'vertical')
            colorbar.scalar_bar_representation.position = [0.85, 0.1]
            file = str(t) + '.png'
            mlab.savefig(file)
            mlab.show()


        elif self.plot_type == 'contour':
            psi = self.simulation.Ψ_plot[index]
            
            L = self.simulation.H.extent/2/Å/L_norm
            Z = self.simulation.H.z_extent/2/Å/Z_norm
            isovalue = np.mean(contrast_vals)

            field = mlab.pipeline.scalar_field(np.abs(psi))

            arr = mlab.screenshot(antialiased = False)

            mlab.outline()
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )
            colour_data = np.angle(psi.T.ravel())%(2*np.pi)
            field.image_data.point_data.add_array(colour_data)
            field.image_data.point_data.get_array(1).name = 'phase'
            field.update()
            field2 = mlab.pipeline.set_active_attribute(field, 
                                                        point_scalars='scalar')
            contour = mlab.pipeline.contour(field2)
            contour.filter.contours= [isovalue,]
            contour2 = mlab.pipeline.set_active_attribute(contour, 
                                                        point_scalars='phase')
            s = mlab.pipeline.surface(contour, colormap='hsv', vmin= 0.0 ,vmax= 2.*np.pi)

            s.scene.light_manager.light_mode = 'vtk'
            s.actor.property.interpolation = 'phong'
            mlab.show()



    def animate_hot(self, L_norm=1, Z_norm=1, unit=milliseconds, time_label_unit='ms', contrast_vals=[0.1, 0.25]):
        """
        Animate a 3D visualization of a scalar field with time label and colorbar.
    
        Parameters:
        - L_norm (float): Normalization factor for x and y dimensions.
        - Z_norm (float): Normalization factor for z dimension.
        - unit (float): Time unit normalization factor.
        - time_label_unit (str): Unit for time label.
        - contrast_vals (list): Minimum and maximum values for contrast normalization.
        """
        mlab.figure(fgcolor = (0.1,0.1,0.1), bgcolor=(0.9, 0.9, 0.9), size=(700, 700))
    
        psi = self.simulation.Ψ[0]
        abs_max = self.simulation.Ψmax
        psi = np.abs(psi / abs_max)
        
        dt_store = self.simulation.total_time / self.simulation.store_steps
    
        L = self.simulation.H.extent / 2 / L_norm
        Z = self.simulation.H.z_extent / 2 / Z_norm
        N = self.simulation.H.N
        Nz = self.simulation.H.Nz
        
        psi = np.clip(psi, contrast_vals[0], contrast_vals[1])
    
        field = mlab.pipeline.scalar_field(psi)
        vol = mlab.pipeline.volume(field)
    
        vol.update_ctf = True
    
        mlab.outline()
        x_latex = '$x/w_o$'
        y_latex = '$y/w_o$'
        z_latex = '$z/\\lambda$'
    
        ax = mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex, nb_labels=6, ranges=(-L, L, -L, L, -Z, Z))
        
        time_label = mlab.text(0.1, 0.9, '', width=0.17, color=(0.1, 0.1, 0.1))  # White text color
        
        #colorbar = mlab.colorbar(orientation='vertical')
        ax.axes.font_factor = 1.9
        ax.label_text_property.font_family = 'times'
        ax.title_text_property.font_family = 'times'
        ax.axes.y_axis_visibility = True
        ax.axes.label_format = '%-#6.2g'
        """
        colorbar = mlab.colorbar(nb_labels=6,orientation = 'vertical')
        colorbar.scalar_bar_representation.position = [0.85, 0.1]
        colorbar_label = colorbar.scalar_bar.label_text_property
        colorbar_label.font_family = 'times'
        colorbar.scalar_bar.label_format = '%.2f'
        colorbar.scalar_bar.unconstrained_font_size = True
        colorbar_label.font_size = 21
        """
        def update_time_label(t):
            time_label.text = '$T = {:.2f}\ {}$'.format(t, time_label_unit)
    
        def update_colorbar(field):
            # Manually update the colorbar range and labels
            colorbar.label_text_property.font_size = 21
            colorbar.data_range = field.mlab_source.scalars.min(), field.mlab_source.scalars.max()
            #colorbar.update_pipeline()
    
        data = {'t': 0.0}
    
        @mlab.animate(delay=10)
        def animation():
            while True:
                data['t'] += 0.05
                k1 = int(data['t']) % self.simulation.store_steps
                psi = self.simulation.Ψ[k1]
    
                psi = np.abs(psi / abs_max)
                psi = np.clip(psi, contrast_vals[0], contrast_vals[1])
    
                field.mlab_source.scalars = psi
                update_time_label(k1 * dt_store / unit)
                mlab.view(azimuth=75, elevation=60, distance=(N + Nz) * 2)
                #update_colorbar(field)
                yield
    
        ua = animation()
        mlab.show()
    

        
    def animate(self, L_norm = 1, Z_norm = 1, unit = milliseconds,time = 'milliseconds', contrast_vals= [0.1, 0.25]):
        #self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        mlab.figure(1,bgcolor = (0,0,0), size=(700, 700))
        


        if self.plot_type == 'abs-volume':
            psi = self.simulation.Ψ[0]
            
            abs_max = self.simulation.Ψmax
            psi = np.abs((psi)/(abs_max))
            
            dt_store = self.simulation.total_time/self.simulation.store_steps


            L = self.simulation.H.extent/2/L_norm
            Z = self.simulation.H.extent/2/Z_norm
            N = self.simulation.H.N
            Nz = self.simulation.H.Nz
            psi = np.where(psi > contrast_vals[1], contrast_vals[1],psi)
            psi = np.where(psi < contrast_vals[0], contrast_vals[0],psi)
            field = mlab.pipeline.scalar_field(psi)
            vol = mlab.pipeline.volume(field)


            # Update the shadow LUT of the volume module.
            vol.update_ctf = True

            mlab.outline()
            x_latex = 'x (mm)'
            y_latex = 'y (mm)'
            z_latex = 'z (mm)'
            
           # x_latex = mlabtex(0.,0.,.0,x_latex)
            #y_latex = mlabtex(0.,0.,.0,x_latex)
           # z_latex = mlabtex(0.,0.,.0,x_latex)
            
        
            mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
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
                    psi = np.where(psi > contrast_vals[1], contrast_vals[1],psi)
                    psi = np.where(psi < contrast_vals[0], contrast_vals[0],psi)
                    
                    
                    


                    field.mlab_source.scalars = psi
                    
                    update_time_label(k1*dt_store/unit)
                    mlab.view(azimuth=60,elevation=60,distance=(N + Nz)*1.5)
                    
                    yield
                    file = str(k1) + '.png'
                    #◘mlab.savefig(file)

            ua = animation()
            mlab.show()
        




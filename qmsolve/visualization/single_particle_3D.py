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


        
        
    def plot2D(self, t, xlim=None, ylim=None, L_norm = 1, Z_norm = 1,unit = femtoseconds, figsize=(15, 15), potential_saturation=0.8, wavefunction_saturation=1.0):


        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("classic")
        """
        fig = plt.figure(figsize=figsize)
        
        ax = fig.add_subplot(1, 1, 1)
        """
        """
        
        sub = 2
        fig, axs = plt.subplots(1,2,figsize = figsize)
        mid = self.simulation.H.N / 2 - 1
        
        #time_axs = axs.text(0.97,0.97, "",  color = "white",transform=axs.transAxes, ha="right", va="top")
        #time_axs.set_text(u"t = {} nanoseconds".format("%.3f"  % (t/unit)))
        tmp = 0
        for i in range(1):
            #print(i)
            z_val = self.simulation.H.particle_system.z[0,0,49]
            axs[i // sub, i % sub].set_xlabel("$x (\mu m)$")
            axs[i // sub, i % sub].set_ylabel("$y (\mu m)$")
            axs[i // sub, i % sub].set_title('$z = 0 \mu m$')
           # axs[i // 3, i % 3].set_title("$\psi(x,y,t)$")





            if xlim != None:
                axs.set_xlim(np.array(xlim)/Å)
            if ylim != None:
                axs.set_ylim(np.array(ylim)/Å)


            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
            index = int((self.simulation.store_steps)/self.simulation.total_time*t)
            
            L = self.simulation.H.extent/Å/L_norm
            #axs[i // sub, i % sub].imshow((self.simulation.H.Vgrid[:,0,:] + self.simulation.Vmin)/(self.simulation.Vmax-self.simulation.Vmin), vmax = 1.0/potential_saturation, vmin = 0, cmap = "gray", origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  

            #axs[i // sub, i % sub].imshow(complex_to_rgba(self.simulation.Ψ_plot[index][:,:,0 + int(i*(100/12))], max_val= wavefunction_saturation), origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  
            tmp = axs[i // sub, i % sub].imshow(abs(self.simulation.Ψ_plot[index][:,:,49])**2,cmap='jet',  extent = [-L/2, L/2, -L/2, L/2])  
            
            #axs[i // sub, i % sub].colorbar()
        strg = (u"t = {} nanoseconds".format("%.3f"  % (t/unit)))
        fig.text(0,0,strg,ha='center',fontsize = 14)
        fig.colorbar(tmp,ax = axs)
        """
        #self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("dark_background")

        fig = plt.figure(figsize=figsize)
        mid = self.simulation.H.N / 2 - 1
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel("$x (\mu m)$")
        ax.set_ylabel("$y (\mu m)$")
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
        #ax.imshow((self.simulation.H.Vgrid + self.simulation.Vmin)/(self.simulation.Vmax-self.simulation.Vmin), vmax = 1.0/potential_saturation, vmin = 0, cmap = "gray", origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  

        tmp = ax.imshow(abs(self.simulation.Ψ_plot[index][:,:,49])**2, cmap='jet', extent = [-L/2, L/2, -L/2, L/2])  
        fig.colorbar(tmp,ax = ax)
        plt.show()


        """
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
        
        L = self.simulation.H.extent/Å/L_norm
        #ax.imshow((self.simulation.H.Vgrid[:,:,0] + self.simulation.Vmin)/(self.simulation.Vmax-self.simulation.Vmin), vmax = 1.0/potential_saturation, vmin = 0, cmap = "gray", origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  

        ax.imshow(complex_to_rgba(self.simulation.Ψ_plot[index][:,:,49], max_val= wavefunction_saturation), origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  
        """
        plt.show()


    def animate2D(self,xlim=None, ylim=None, figsize=(7, 7),L_norm = 1, unit = femtoseconds, animation_duration = 5, fps = 20, save_animation = False, potential_saturation=0.8, wavefunction_saturation=0.8):
        total_frames = int(fps * animation_duration)
        dt = self.simulation.total_time/total_frames
        self.simulation.Ψ_plot = self.simulation.Ψ/self.simulation.Ψmax
        plt.style.use("dark_background")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        index = 0
        
        L = self.simulation.H.extent/Å/L_norm
        potential_plot = ax.imshow((self.simulation.H.Vgrid[:,0,:] + self.simulation.Vmin)/(self.simulation.Vmax-self.simulation.Vmin), vmax = 1.0/potential_saturation, vmin = 0, cmap = "gray", origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  
        wavefunction_plot = ax.imshow(complex_to_rgba(self.simulation.Ψ_plot[0][:,:,49], max_val= wavefunction_saturation), origin = "lower", interpolation = "bilinear", extent=[-L / 2,L / 2,-L / 2,L / 2])


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
        time_ax.set_text(u"t = {} nanoseconds".format("%.3f"  % 0.00))


        #print(total_frames)
        animation_data = {'t': 0.0, 'ax':ax ,'frame' : 0}
        def func_animation(*arg):
            
            time_ax.set_text(u"t = {} nanoseconds".format("%.3f"  % (animation_data['t']/unit)))

            animation_data['t'] = animation_data['t'] + dt
            if animation_data['t'] > self.simulation.total_time:
                animation_data['t'] = 0.0

            #print(animation_data['frame'])
            animation_data['frame'] +=1
            index = int((self.simulation.store_steps)/self.simulation.total_time * animation_data['t'])

            wavefunction_plot.set_data(complex_to_rgba(self.simulation.Ψ_plot[index][:,:,49], max_val= wavefunction_saturation))
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


    def plot3D(self, t, L_norm = 1, Z_norm = 1,figsize=(7, 7),unit = femtoseconds, contrast_vals= [0.1, 0.25]):

        mlab.figure(bgcolor=(0,0,0), size=(700, 700))
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
        psi = self.simulation.Ψ_plot[index]
        psi = np.abs(psi)**2
         
        # print(psi)
        
        L = self.simulation.H.extent/2/Å/L_norm
        Z = self.simulation.H.z_extent/2/Å/Z_norm
        N = self.simulation.H.N
        
        #surf = mlab.mesh(self.simulation.H.particle_system.x[:,:,0],self.simulation.H.particle_system.y[:,:,0],psi[:,:,49],colormap='jet')
        surf = mlab.surf(psi[:,:,49],warp_scale="auto",colormap='jet')
        mlab.colorbar(surf,orientation='vertical')  
         
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
         #mlab.axes(xlabel='x [Å]', ylabel='y [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
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
        if self.plot_type == 'volume':
            
            psi = self.simulation.Ψ_plot[index]
            
            L = self.simulation.H.extent/2/Å/L_norm
            Z = self.simulation.H.z_extent/2/Å/Z_norm

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
            mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
            mlab.show()


        if self.plot_type == 'abs-volume':
            
            psi = self.simulation.Ψ_plot[index]
            
            
           # print(psi)

            L = self.simulation.H.extent/2/Å/L_norm
            Z = self.simulation.H.z_extent/2/Å/Z_norm
            N = self.simulation.H.N

            vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.abs(psi)[:,:,49]), vmin= contrast_vals[0], vmax= contrast_vals[1])          
            
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
        
        
    def animate(self, L_norm = 1, Z_norm = 1, unit = femtoseconds,time = 'femtoseconds', contrast_vals= [0.1, 0.25]):
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
                    mlab.view(azimuth=60,elevation=60,distance=N*3.5)
                    
                    yield
                    file = str(k1) + '.png'
                    #◘mlab.savefig(file)

            ua = animation()
            mlab.show()
        elif self.plot_type == 'contour':
            psi = self.simulation.Ψ[0]
            L = self.simulation.H.extent/2/Å/L_norm
            Z = self.simulation.H.z_extent/2/Å/Z_norm
            N = self.simulation.H.N
            isovalue = np.mean(contrast_vals)

            dt_store = self.simulation.total_time/self.simulation.store_steps
            abs_max= self.simulation.Ψmax
            psi = (psi)/(abs_max)

            field = mlab.pipeline.scalar_field(np.abs(psi))

            arr = mlab.screenshot(antialiased = False)

            mlab.outline()
            #mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]',nb_labels=6 , ranges = (-L,L,-L,L,-L,L) )
            
            x_latex = '$x/w_o$'
            y_latex = '$y/w_o$'
            z_latex = '$z/\lambda$'
            
           # x_latex = mlabtex(0.,0.,.0,x_latex)
            #y_latex = mlabtex(0.,0.,.0,x_latex)
           # z_latex = mlabtex(0.,0.,.0,x_latex)
            
        
            mlab.axes(xlabel=x_latex, ylabel=y_latex, zlabel=z_latex,nb_labels=6 , ranges = (-L,L,-L,L,-Z,Z) )
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
            time_label = mlab.text(0.1, 0.9, '', width=0.3, color=(1, 1, 1))  # White text color
            def update_time_label(t):
                time_label.text = 'Time: {:.2f} {}'.format(t, unit)

            data = {'t': 0.0}
            @mlab.animate(delay=10)
            def animation():
                while (1):
                    data['t'] += 0.05
                    k1 = int(data['t']) % (self.simulation.store_steps)

                    psi = self.simulation.Ψ[k1]*np.exp( 1j*2*np.pi/10*k1)
                    psi = (psi)/(abs_max)
                    np.copyto(colour_data, np.angle(psi.T.ravel())%(2*np.pi))
                    field.mlab_source.scalars = np.abs(psi)
                    
                    update_time_label(k1*dt_store/unit)
                    
                    yield
                    file = str(k1) + '.png'
                    #mlab.savefig(file)
            ua = animation()
            mlab.show()
        




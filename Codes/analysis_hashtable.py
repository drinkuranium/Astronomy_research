# 2020-02-07, Dong Uk Kim, add_gas.py
# This program analyzes HDF5 output file by creating hash table

import numpy as np
import matplotlib.pyplot as plt
import sys
import time  ## For debuging purpose
from load_from_snapshot import load_from_snapshot
from pathlib import Path


def load_sorted_data(value, sdir, snum, ptype_list=[0,1,2,3,4,5]):
    """This subrutine load the densinated data (by value) of the desinated
    particles (by ptype_list) in particle ID sorted order.
    
    Arguments:    
      value (string): the value to extract from the HDF5 file. this is a string
        with the same name as in the HDF5 file. if you arent sure what those
        values might be, setting value to 'keys' will return a list of all the
        HDF5 keys for the chosen particle type, or 'header_keys' will return
        all the keys in the header.
        (example: 'Time' returns the simulation time in code units (single scalar).
        'Coordinates' will return the [x,y,z] coordinates in an [N,3] matrix 
        for the N resolution elements of the chosen type)
             
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

    Optionals:
      ptype_list (int list or int): Specify particle types that whose data
        will be lodaed. Default load data of every particle types. Users can
        specify particle types by passing an array. Data of every particle 
        types included in the array will be loaded. If users want to plot 
        only one ptype, then entering just int number is okay.

    Returns:
      ptype_list_final (int list): Store ptypes which exist in the data.

      data_list (object ndarray): Store data corresponds to the 'value' in each
        element. Particle types are given in ptype_list_final. Particles are
        sorted in ID manner.
    """
    
    ## Make int ptype into array
    if type(ptype_list) == int: ptype_list = [ptype_list]

    ## Load data of desinated particle types
    data_list = np.array([0,0,0,0,0,0], dtype=object)
    ptype_list_final = ptype_list[:]
    for ptype in ptype_list:
        data = load_from_snapshot(value, ptype, sdir, snum)
        ## When desinated ptype particle does not exist
        if type(data) == int and data == 0:
            print("ptype %i particle dose not exist in the galaxy." %ptype)
            ptype_list_final.remove(ptype)
        ## Store data into data_list, which will be returned
        else:
            ## Sort particles in ID order
            pid = load_from_snapshot('ParticleIDs', ptype, sdir, snum)
            sorted_data = data[np.argsort(pid)]
            data_list[ptype] = sorted_data

    ## Immediately exit program if there are no particles of desniated ptypes
    if len(ptype_list_final) == 0:
        sys.exit("Particles of desinated ptypes do not exist at all. Change ptypes.")
    
    ## Select data of only existing particles
    data_list = data_list[ptype_list_final]

    ## Make correction to postion and velocity data so that
    ## Position: the galactic center (center of mass) is located at (0,0).
    ## Velocity: the net momentum becomes 0.
    if value == 'Coordinates' or value == 'Velocities':
        m_total = 0
        cm = 0
        pnum = load_from_snapshot('NumPart_Total', 0, sdir, snum)
        ptype_exist = np.arange(0, 6, 1, dtype=int)[np.where(pnum>0)]
        for ptype in ptype_exist:
            ##Negelect halo particle, they might be affect cm large while
            ##they merely affects on the galactic center
            if ptype == 1:
                continue
            m = load_from_snapshot('Masses', ptype, sdir, snum)
            m_total += sum(m)
            temp_data = load_from_snapshot(value, ptype, sdir, snum)
            cm += np.dot(m, temp_data)
        cm /= m_total
        for i in range(len(data_list)):
            data_list[i] -= cm

    return ptype_list_final, data_list


class Particle:
    """Represents a particle.
    
    Attributes:
      pid (int): ID of the particle
      ploc (1D float ndarray): Location of the particle
      next (Particle): Points to the next particle in a linked list
    """

    def __init__(self, pid, ploc):
        """Set particle id (pid) and location (ploc)"""
        self.id = pid
        self.loc = ploc
        self.next = None


class HashTables:
    """Stores particle information as a form of a hash table.
    
    Attributes:
      size (float): Size of a box that particles will be stored
      grid_num (int): Number of grids, decides the number of table elements
      grid_len (float): Length of each grid
      data (4D Particle ndarray): Actual storage of particle informaion.
        Shape is (len(ptype_list), grid_num, grid_num, grid_num). Each element
        of the data stores specific ptype particles in 3D Particle ndarray.
        Each elements of 3D particle ndarray acts as an linked list of class
        'Particle'.
      NumPart (4D int ndarray): Stores the number of particles in each element
        of data in corresponding element.
    """

    def __init__(self, sdir, snum, size=10, grid_num=50):
        """Create list of hash table that stores particle data. 
        Each element of the class 'HashTable' stores a hash table of particles.
        Then each element of hash tables stores the data of particles as
        linked lists of class 'Particle'. The location of each element
        of hash table corresponds to the location of particles.

        Parameters:
          sdir (string): parent directory of the snapshot file or immediate
            snapshot sub-directory if it is a multi-part file
                
          snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
            Note for multi-part files, this is just the number of the 'set',
            i.e. if you have snapshot_001.N.hdf5, set this to '1',
            not 'N' or '1.N'
    
        Optionals:
          ptype_list (int list or int): Specify particle types that whose data
            will be lodaed. Default load data of every particle types. You can
            specify particle types by passing an array. Data of every particle 
            types included in the array will be loaded. If you want to plot 
            only one ptype, then entering just int number is okay.

          size (float): Size of a box that particles will be stored in kpc.
            Default value is 10, which is good for examine bulge and disk.

          grid_num (int): Number of grids, decides the number of table element.
            Default value is 50, but optimal value depends on the resolution
            of the simulation (particle number of the simulation).
        """

        self.size = size  ## Size of box which will be included in the table
        self.grid_num = grid_num   ## Number of table elements
        self.grid_len = size*2/grid_num  ## Grid length
        self.data = np.empty((6, grid_num, grid_num, grid_num), dtype=Particle)  ## Actual data storage
        self.NumPart = np.zeros_like(self.data, dtype=int)  ## Number of particle in each table elements

        ## load data
        ptype_exist, plocs = load_sorted_data('Coordinates', sdir, snum)
        ptype_exist, pids = load_sorted_data('ParticleIDs', sdir, snum)
        pnums = load_from_snapshot('NumPart_Total', 0, sdir, snum)
        self.ptype_exist = ptype_exist

        ## Assign particles to the hash table data
        for i in range(len(ptype_exist)):
            ploc, pid, ptype = plocs[i], pids[i], ptype_exist[i]
            pnum = pnums[ptype]
            for j in range(pnum):
                part = Particle(pid[j], ploc[j])
                if max(abs(part.loc)) > size:
                    del part
                    continue
                else:
                    self.add_particle(part, ptype)
            
    def add_particle(self, NewParticle, ptype):
        """Add new particle to the hash table (data) as a form of linked list.

        Parameters:
          NewParticle (Particle): The particle that will be added.
          ptype (int): The type of the particle
        """

        temp_loc = NewParticle.loc + self.size
        loc_index = np.array(temp_loc//self.grid_len, dtype=int)
        index = tuple(np.append(ptype, loc_index))
        if self.data[index] == None:
            self.data[index] = NewParticle
            self.NumPart[index] += 1
        else:
            NewParticle.next = self.data[index]
            self.data[index] = NewParticle
            self.NumPart[index] += 1
        return None

    def search_particle(self, pid):
        """Search for a particle of specific id in the hash table.

        Parameters:
          pid (int): ID of the target particle

        Returns:
          CheckParticle (Particle): The particle of desinated ID
            or None: When the particle of desinated ID does not exist
        """

        for ptype in range(6):
            for x in range(self.grid_num):
                for y in range(self.grid_num):
                    for z in range(self.grid_num):
                        if self.data[ptype,x,y,z] == None:
                            continue
                        else:
                            CheckParticle = self.data[ptype,x,y,z]
                            while CheckParticle.next != None:
                                if CheckParticle.id == pid:
                                    return CheckParticle
                                else:
                                    CheckParticle = CheckParticle.next
                            if CheckParticle.id == pid:
                                return CheckParticle
        ## When the particle does not exist
        return None

    def plot_particles(self, trackID=0, ptype_list=[0,1,2,3,4,5], 
                       axes=[(0,1), (0,2), (1,2)], save=False):
        """Plots the particles of desinated ptypes in the hash table onto
        desinated projected planes. Also it can save figures in the desinated
        directory.

        Optionals:
          track_ID (int): The ID of the particle that you want to emphasize.
            Default is 0, which will not emphasize any particle.

          ptype_list (int list or int): Specify particle types that will be
            plotted. Default will plot every existing components.
            You can specify particle types by passing an array.
            Every particle types included in the array will be plotted.
            Just passing a int number is also okay,
            it will be treated as a list of length 1.
            
          axes (tuple list or tuple): Specify the projected plane. Users
            can specify the projected plane by passing a list of tuple
            (or just a tuple) of combination of 0 (for x-axis), 1 (for y-axis)
            or 2 (for z-axis). For example, if one want to choose xy-plane,
            then pass axis=(0,1). Other values will give error and immediate
            return of this function. Default will plot all (xy,xz,yz) plane.

          save (boolean): Determine whether save the plot or not.
            Default is False. You must specify the saving directory.
            Default will be a sub-directory of the data directory,
            and if the directory does not exist, then it will create it.
        """
        
        ## Processing input parameters
        ## Examine the existence of ptype
        for ptype in ptype_list:
            if ptype < 0 or ptype > 5 or type(ptype) != int:
                print("Please enter valid ptypes.")
                return None
            if not ptype in self.ptype_exist:
                ptype_list.remove(ptype)
        ## Treat tuple as a list of length 1
        if type(axes) == tuple: axes = [axes]
        ## Inspect legitimacy of the format
        for axis in axes:
            if axis[0] != 0 and axis[0] !=1 and axis[0] != 2 and \
            axis[1] != 0 and axis[1] !=1 and axis[1] != 2:
                print("Please follow the format: [0, 1, 2] for [x, y, z] axis \
                      - then specify the plane by pairing axis like (0,2), \
                      in case of yz-plane.")
                return None 
    
        ## Plot galaxies within given size
        axis_label = ["X", "Y", "Z"]
        for x_axis, y_axis in axes:
            ## Set ploting varaibles
            plt.figure(figsize = (10,10))
            plt.axis('equal')
            markers = ['c.', 'k.', 'r.', 'b.', 'm.', 'ko']
            markers_size = 2/self.size
            if markers_size < 0.1:
                markers_size = 0.1
            for ptype in ptype_list:
                ## Construct 2D array of locations of particles
                ploc = np.array([0,0,0])  ## Assign dummy for initialization
                exist = np.where(self.data[ptype] != None)  ## Choose only where particles exist
                for linked_list in self.data[ptype][exist]:
                    while not linked_list.next == None:
                        ploc = np.vstack((ploc, linked_list.loc))
                        linked_list = linked_list.next
                    ploc = np.vstack((ploc, linked_list.loc))
                ploc = np.delete(ploc, 0, axis=0)  ## Remove dummy
                ## Plot particles
                n = 1  ## Increase n to prevent overwriting, with cost of time
                for i in range(n):
                    data_len = len(ploc)
                    start = int(round(data_len*i/n))
                    end = int(round(data_len*(i+1)/n))
                    plt.plot(ploc[start:end,x_axis], ploc[start:end,y_axis],
                             markers[ptype], markersize=markers_size)
            ## Emphasizes desinated particle
            if trackID != 0:
                Target = self.search_particle(trackID)
                if Target == None:
                    print("No particle of desinated ID in hash table. \
                          Particle ID might not be valid or particle is out \
                          of the box. Please change the particle ID.")
                else:
                    plt.plot(Target.loc[x_axis], Target.loc[y_axis], 'k*',
                             markersize=10)
            ## Set plotting variables
            x_label, y_label = axis_label[x_axis], axis_label[y_axis]
            plt.xlabel(x_label+" axis [kpc]")
            plt.ylabel(y_label+" axis [kpc]")
            plt.title("t=%.2f Gyr, projected onto %s%s-plane galaxy" %(snum*0.01, x_label, y_label))
            plt.xlim((-self.size, self.size))
            plt.ylim((-self.size, self.size))
            ## Plot the origin
            plt.plot(0, 0, 'ko', markersize=5)
            
            ## Save the figure if save flag is on
            if save:
                ## Create the directory if desniated directory does not exist
                Path(sdir+"/plot/galaxy_image/%s%s-plane"
                     %(x_label, y_label)).mkdir(parents=True, exist_ok=True)
                plt.savefig(sdir+"/plot/galaxy_image/%s%s-plane/t=%.2f Gyr particle plot.png"
                            %(x_label, y_label, snum*0.01))
            plt.show()
        return None

    def plot_density_contour(self, ptype_list=[0,1,2,3,4,5],
                              axes=[(0,1),(0,2),(1,2)], save=False):
        """Plots density contours of the particles of type desinated by
        ptype_list, within specified size (by sizes) onto specified projected
        plane (by axes). Can save the each figure to sub-directory of the data
        directory (specified by save)

        Optionals:
          ptype_list (int list or int): Specify particle types that will be
            plotted. Default will plot every existing components.
            You can specify particle types by passing an array.
            Every particle types included in the array will be plotted.
            Just passing a int number is also okay,
            it will be treated as a list of length 1.
            
          axes (tuple list or tuple): Specify the projected plane. Users
            can specify the projected plane by passing a list of tuple
            (or just a tuple) of combination of 0 (for x-axis), 1 (for y-axis)
            or 2 (for z-axis). For example, if one want to choose xy-plane,
            then pass axis=(0,1). Other values will give error and immediate
            return of this function. Default will plot all (xy,xz,yz) plane.

          save (boolean): Determine whether save the plot or not.
            Default is False. You must specify the saving directory.
            Default will be a sub-directory of the data directory,
            and if the directory does not exist, then it will create it.
        """

        ## Processing input parameters
        ## Examine the existence of ptype
        for ptype in ptype_list:
            if ptype < 0 or ptype > 5 or type(ptype) != int:
                print("Please enter valid ptypes.")
                return None
            if not ptype in self.ptype_exist:
                ptype_list.remove(ptype)
        ## Treat tuple as a list of length 1
        if type(axes) == tuple: axes = [axes]
        ## Inspect legitimacy of the format
        for axis in axes:
            if axis[0] != 0 and axis[0] !=1 and axis[0] != 2 and \
            axis[1] != 0 and axis[1] !=1 and axis[1] != 2:
                print("Please follow the format: [0, 1, 2] for [x, y, z] axis \
                      - then specify the plane by pairing axis like (0,2), \
                      in case of yz-plane.")
                return None 

        ## Plot density contour
        axis_label = ["X", "Y", "Z"]
        grid_volume = self.grid_len**3
        x_grid = np.linspace(-self.size, self.size, self.grid_num, endpoint=True)
        y_grid = np.linspace(-self.size, self.size, self.grid_num, endpoint=True)
        X, Y = np.meshgrid(x_grid, y_grid)
        for x_axis, y_axis in axes:  ## Plot for every projected planes
            density = np.zeros((self.grid_num, self.grid_num))
            for ptype in self.ptype_exist:
                summing_axis = 3-x_axis-y_axis
                projected_data = np.sum(self.NumPart[ptype], axis=summing_axis)
                density += projected_data
            density = density.T/grid_volume
            ## Plot contour
            ## Choose this when you do not know the exact scale
            fig, ax = plt.subplots(figsize=(12,10))
            plt.axis('equal')
            contour = plt.contourf(X, Y, density, 40)
            fig.colorbar(contour)
            ## Choose this when you know exact scale and need to compare it
            ## with the results from different snapshots
#            plt.figure(figsize=(10,10))
#            plt.axis('equal')
#            plt.contourf(X, Y, density, levels=np.arange(0, 1500, 25))
            x_label, y_label = axis_label[x_axis], axis_label[y_axis]
            plt.xlabel(x_label+" axis [kpc]")
            plt.ylabel(y_label+" axis [kpc]")
            plt.xlim((-self.size, self.size))
            plt.ylim((-self.size, self.size))
            plt.title("t=%.2f Gyr, Galaxy density contour projected onto %s%s-plane"
                      %(snum*0.01, x_label, y_label))
            ## Save the figure if save flag is on
            if save:
                ## Create the directory if desniated directory does not exist
                Path(sdir+"/plot/density_contour/%s%s-plane"
                     %(x_label, y_label)).mkdir(parents=True, exist_ok=True)
                plt.savefig(sdir+"/plot/density_contour/%s%s-plane/t=%.2f Gyr particle plot.png"
                            %(x_label, y_label, snum*0.01))
            plt.show()
    
        return None   

    def __del__(self):
        """Deallocate every momery assigned to this class.
        
        I am not sure that this destructor is essential, since Python has
        garbage collector. But sometimes I found that sometimes memory leaks
        occur in serious degree, so I decide to add this feature.
        """

        for ptype in range(6):
            for x in range(self.grid_num):
                for y in range(self.grid_num):
                    for z in range(self.grid_num):
                        if self.data[ptype,x,y,z] == None:
                            continue
                        else:
                            NextParticle = self.data[ptype,x,y,z]
                            while NextParticle.next != None:
                                CurrentParticle = NextParticle
                                NextParticle = CurrentParticle.next
                                del CurrentParticle
                            del NextParticle


## Select model
sdir = "/home/du/gizmo/TestGas/Test_Gas_Generation/results"
#snum = 10
#hash_tables = HashTables(sdir, snum, grid_num=51)
#hash_tables.plot_particles(ptype_list=[2,1,0,4], axes=(0,1), trackID=200500)
#hash_tables.plot_density_contour(ptype_list=[0,1,2,4], axes=(0,1))
#del hash_tables

snum_list = np.arange(0, 40, 1)
for snum in snum_list:
    hash_tables = HashTables(sdir, snum, grid_num=51)
    hash_tables.plot_particles(ptype_list=[2,1,0,4], axes=(0,1), trackID=200500)   
    del hash_tables
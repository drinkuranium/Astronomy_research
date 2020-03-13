## 2020-03-12, Dong Uk Kim, plot_density_contour_before_opt.py
## This code file stores 'initial version' of implementations of
## plotting density contours using different Kernel functions
## (Gaussian, Cubic spline).


import numpy as np
import matplotlib.pyplot as plt
import math
import time  ## For debugging purpose
from pathlib import Path
from load_from_snapshot import load_from_snapshot
from analysis import load_data


## In analysis.py I optimized this function (although fundamental algorithm
## is the smae). It is about 0.25*grid_num times faster after optimization
## when grid_num is bigger than 100 and contour includes halo particles.
## When the contour does not incldue halo particles, about 2.5 times faster.
## I just saved the priomoridal version of the function for future reference.
def cubic_spline_kernel(sdir, snum, ptype_list=[0,1,2,3,4,5],
                        sizes=[10.], axes=[(0,1),(0,2),(1,2)], save=False):
    """This subrutine plots density contours of the particles of type desinated
    by ptype_list, within specified size (by sizes) onto specified projected
    plane (by axes). Can save the each figure to sub-directory of the data
    directory (specified by save). Uses cubic spline kernel function.
    
    Arguments:
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file.
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

    Optionals:
      sizes (int, float list or int, float): The size of the plot in code unit
        (nomarlly kpc - but inspect your setting!). Default value is 10,
        recommended for observing disk and bulge structures. If an user
        specifies this by passing a float number list, then it will only
        repeatedly plot the density contour within the size specified by each
        element of the list. Just passing a float number is also okay, it will
        be treated as a list of length 1.
        Special value -1 - then the size of the plot will be scaled to
        incorporate every particles.
    
      axes (tuple list or tuple): Specify the projected plane. Users
        can specify the projected plane by passing a list of tuple (or just 
        a tuple) of combination of 0 (for x-axis), 1 (for y-axis) or 2
        (for z-axis). For example, if one want to choose xy-plane, then pass
        axis=(0,1). Other values will give error and immediate return of this
        function. Default will plot all (xy,xz,yz) plane.
    
      ptype_list (int list or int): Specify particle types that will be
        plotted. Default will plot every existing components. Users can specify
        particle types by passing an array. Every particle types included in
        the array will be plotted. Just passing a int number is also okay,
        it will be treated as a list of length 1.
    
      save (boolean): Determine whether save the plot or not. Default is False.
        An user must specify the saving directory. Default will be a
        sub-directory of the data directory, and if the directory does not
        exist, then it will create it.

    Returns: Nothing
    """

    ## Load the locations of the particles
    value = 'Coordinates'
    ptype_exist, data_list = load_data(value, sdir, snum, ptype_list)
    ## Load mass data
    value = 'Masses'
    masses = np.zeros(6)
    for ptype in ptype_exist:
        mass = load_from_snapshot(value, ptype, sdir, snum)
        masses[ptype] = mass[0]  ## Assume mass of each ptype particle is the same

    ## Treat float of int number as a list of length 1
    if type(sizes) == float or type(sizes) == int: sizes = [sizes]
    
    ## Inspect legitimacy of size
    for size in sizes:
        if size<=0:
            print("Non-positive size is entered. Please enter positive value for the size.")

    ## Treat tuple as a list of length 1
    if type(axes) == tuple: axes = [axes]

    ## Inspect legitimacy of the format
    for axis in axes:
        if axis[0] != 0 and axis[0] !=1 and axis[0] != 2 and \
        axis[1] != 0 and axis[1] !=1 and axis[1] != 2:
            print("Please follow the format: [0, 1, 2] for [x, y, z] axis - then \
                  specify the plane by pairing axis like (0,2), in case of yz-plane.")
            return None 

    ## Plot density contour
    axis_label = ["X", "Y", "Z"]
    for size in sizes:

        ## Create meshgrid to plot contour
        grid_num = 401
        x_grid = np.linspace(-size, size, grid_num, endpoint=True)
        y_grid = np.linspace(-size, size, grid_num, endpoint=True)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_len = 2*size/(grid_num-1)

        for x_axis, y_axis in axes:       

            ## Density of each grid
            density = np.zeros((grid_num, grid_num))
            
            ## Remove data out of the contour domain 
            for i in range(len(ptype_exist)):
                ptype, data = ptype_exist[i], data_list[i]
                ## For Kernel Estimation
                sigma = np.std(data[:,[x_axis,y_axis]], axis=0)
                h = sigma*len(data)**(-1/6)  ## Silverman's rule in 2D
                ## Arbitrarily decrease bandwidth
                ## For observing central dense region
                resolution = 2  ## Arbitrary number!
                h /= resolution
                h1, h2 = h[x_axis], h[y_axis]
                index = np.where((abs(data[:,x_axis])<size) & (abs(data[:,y_axis])<size))
                data = data[index]
                ## Construct matrix that stores the value of kernel function
                ## Kernel function: Cubic spline
                kernel_len_x = int(2*h1/grid_len)
                kernel_len_y = int(2*h2/grid_len)
                print('ptype %d kernel length: %d, %d' %(ptype, kernel_len_x, kernel_len_y))
                
#                 I try to use numpy module as much as possible for performance
#                 First construct 3 matrices which stores the value of cubic
#                 spline kernel in each domain
#                 In this code, kernel_temp[0]: for 0<=q<1
#                               kernel_temp[1]: for 1<=q<2
#                               kernel_temp[2]: for 2<=q
#                 Then pile up them into 3D array and sort each element and
#                 pick medium value - then it is cubic spline kernel.
#                kernel_temp = np.zeros((3, kernel_len_x*2+1, kernel_len_y*2+1))
#                for j in range(len(kernel_temp[0])):
#                    for k in range(len(kernel_temp[0,j])):
#                        kernel_temp[0,j,k] = (((j-kernel_len_x)/h1)**2
#                                              + ((k-kernel_len_y)/h2)**2)**(1/2)
#                kernel_temp[0] *= grid_len
#                kernel_temp[1] = 4*(2-kernel_temp[0])**3
#                kernel_temp[0] = 1-3/2*kernel_temp[0]**2+3/4*kernel_temp[0]**3
#                kernel_temp = np.sort(kernel_temp, axis=0)
#                kernel = kernel_temp[1]
                kernel = np.zeros((kernel_len_x*2+1, kernel_len_y*2+1))
                for j in range(len(kernel)):
                    for k in range(len(kernel[0])):
                        temp = grid_len*(((j-kernel_len_x)/h1)**2
                                         + ((k-kernel_len_y)/h2)**2)**(1/2)
                        if temp > 2:
                            kernel[j,k] = 0
                        elif temp > 1:
                            kernel[j,k] = (2-temp)**3
                        else:
                            kernel[j,k] = 4*(1-3/2*temp**2+3/4*temp**3)
              
                ## Assign particles to grid points and construct density
                data += size + size/grid_num/2
                for particle in data:
                    grid_x = int(particle[x_axis]//grid_len)
                    grid_y = int(particle[y_axis]//grid_len)
                    ## Boundaries
                    x_left = min(grid_x, kernel_len_x)
                    x_right = min(grid_num-grid_x-1, kernel_len_x)
                    y_left = min(grid_y, kernel_len_y)
                    y_right = min(grid_num-grid_y-1, kernel_len_y)
                    sgrid = np.copy(kernel[kernel_len_x-x_left : kernel_len_x+x_right+1,
                                           kernel_len_y-y_left : kernel_len_y+y_right+1])
                    sgrid /= np.sum(kernel)  ## Normalization for conservation
                    sgrid *= masses[ptype]/grid_len**3
                    density[grid_x-x_left : grid_x+x_right+1,
                            grid_y-y_left : grid_y+y_right+1] += sgrid
            density = density.T        

            ## Plot contour
            fig, ax = plt.subplots(figsize=(12,10))
            plt.axis('equal')
            contour = plt.contourf(X, Y, density, 40)
            fig.colorbar(contour)
#            plt.figure(figsize=(10,10))
#            plt.axis('equal')
#            plt.contourf(X, Y, density, levels=np.arange(0, 1500, 25))
#            x_label, y_label = axis_label[x_axis], axis_label[y_axis]
            x_label, y_label = axis_label[x_axis], axis_label[y_axis]
            plt.xlabel(x_label+" axis [kpc]")
            plt.ylabel(y_label+" axis [kpc]")
            plt.xlim((-size, size))
            plt.ylim((-size, size))
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


def Gaussian_kernel(sdir, snum, ptype_list=[0,1,2,3,4,5],
                    sizes=[10.], axes=[(0,1),(0,2),(1,2)], save=False):
    """This subrutine plots density contours of the particles of type desinated
    by ptype_list, within specified size (by sizes) onto specified projected
    plane (by axes). Can save the each figure to sub-directory of the data
    directory (specified by save). Uses Gaussian kernel function.
    
    Arguments:
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file.
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

    Optionals:
      sizes (int, float list or int, float): The size of the plot in code unit
        (nomarlly kpc - but inspect your setting!). Default value is 10,
        recommended for observing disk and bulge structures. If an user
        specifies this by passing a float number list, then it will only
        repeatedly plot the density contour within the size specified by each
        element of the list. Just passing a float number is also okay, it will
        be treated as a list of length 1.
        Special value -1 - then the size of the plot will be scaled to
        incorporate every particles.
    
      axes (tuple list or tuple): Specify the projected plane. Users
        can specify the projected plane by passing a list of tuple (or just 
        a tuple) of combination of 0 (for x-axis), 1 (for y-axis) or 2
        (for z-axis). For example, if one want to choose xy-plane, then pass
        axis=(0,1). Other values will give error and immediate return of this
        function. Default will plot all (xy,xz,yz) plane.
    
      ptype_list (int list or int): Specify particle types that will be
        plotted. Default will plot every existing components. Users can specify
        particle types by passing an array. Every particle types included in
        the array will be plotted. Just passing a int number is also okay,
        it will be treated as a list of length 1.
    
      save (boolean): Determine whether save the plot or not. Default is False.
        An user must specify the saving directory. Default will be a
        sub-directory of the data directory, and if the directory does not
        exist, then it will create it.

    Returns: Nothing
    """

    ## Load the locations of the particles
    value = 'Coordinates'
    ptype_exist, data_list = load_data(value, sdir, snum, ptype_list)
    ## Load mass data
    value = 'Masses'
    masses = np.zeros(6)
    for ptype in ptype_exist:
        mass = load_from_snapshot(value, ptype, sdir, snum)
        masses[ptype] = mass[0]  ## Assume mass of each ptype particle is the same

    ## Treat float of int number as a list of length 1
    if type(sizes) == float or type(sizes) == int: sizes = [sizes]
    
    ## Inspect legitimacy of size
    for size in sizes:
        if size<=0:
            print("Non-positive size is entered. Please enter positive value for the size.")

    ## Treat tuple as a list of length 1
    if type(axes) == tuple: axes = [axes]

    ## Inspect legitimacy of the format
    for axis in axes:
        if axis[0] != 0 and axis[0] !=1 and axis[0] != 2 and \
        axis[1] != 0 and axis[1] !=1 and axis[1] != 2:
            print("Please follow the format: [0, 1, 2] for [x, y, z] axis - then \
                  specify the plane by pairing axis like (0,2), in case of yz-plane.")
            return None 

    ## Plot density contour
    axis_label = ["X", "Y", "Z"]
    for size in sizes:

        ## Create meshgrid to plot contour
        grid_num = 51
        x_grid = np.linspace(-size, size, grid_num, endpoint=True)
        y_grid = np.linspace(-size, size, grid_num, endpoint=True)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_len = 2*size/(grid_num-1)

        for x_axis, y_axis in axes:       

            ## Density of each grid
            density = np.zeros((grid_num, grid_num))
            
            ## Remove data out of the contour domain 
            for i in range(len(ptype_exist)):
                ptype, data = ptype_exist[i], data_list[i]
                ## For Kernel Estimation
                sigma = np.std(data[:,[x_axis,y_axis]], axis=0)
                h = sigma*len(data)**(-1/6)  ## Silverman's rule in 2D
                index = np.where((abs(data[:,x_axis])<size) & (abs(data[:,y_axis])<size))
                data = data[index]
                ## Construct kernel matrix
                cutoff_intensity = 0.1
                ## Kernel should include every elements where the elements
                ## are bigger than 0.1 * origin value
                kernel_len = math.ceil(max(h)/grid_len*(-2*np.log(cutoff_intensity))**(1/2))
                kernel = np.ones((kernel_len*2+1, kernel_len*2+1))
                origin = kernel_len
                print(np.shape(kernel))
                for j in range(origin):
                    kernel[j] *= np.exp(-((j-origin)*grid_len/h[0])**2)
                kernel[origin+1:] = np.flip(kernel[:origin], axis=0)
                for j in range(origin):
                    kernel[:,j] *= np.exp(-((j-origin)*grid_len/h[1])**2)
                kernel[:,origin+1:] = np.flip(kernel[:,:origin], axis=1)
                kernel /= np.sum(kernel)  ## Normalization
                ## Decide limit of grid that will be included in the kernel
                ## When the value of kernel function at a grid is smaller than
                ## kernel function at origin * cutoff_intensity, do not include
                cutoff_x = math.ceil(h[0]/grid_len*(-2*np.log(cutoff_intensity))**(1/2))
                cutoff_y = math.ceil(h[1]/grid_len*(-2*np.log(cutoff_intensity))**(1/2))
                if cutoff_x > grid_num: cutoff_x = grid_num
                if cutoff_y > grid_num: cutoff_y = grid_num
                ## Assign particles to grid points and construct density
                print(cutoff_x, cutoff_y)
                data += size + size/grid_num/2
                for particle in data:
                    grid_x = int(particle[x_axis]//grid_len)
                    grid_y = int(particle[y_axis]//grid_len)
                    ## Boundaries
                    x_left = min(grid_x, cutoff_x)
                    x_right = min(grid_num-grid_x-1, cutoff_x)
                    y_left = min(grid_y, cutoff_y)
                    y_right = min(grid_num-grid_y-1, cutoff_y)
                    sgrid = np.copy(kernel[origin-x_left : origin+x_right+1,
                                           origin-y_left : origin+y_right+1])
                    sgrid_no_boundary = np.copy(kernel[origin-cutoff_x:origin+cutoff_x+1,
                                                       origin-cutoff_y:origin+cutoff_y+1])
                    sgrid /= np.sum(sgrid_no_boundary)  ## Normalization for conservation
                    sgrid *= masses[ptype]/grid_len**3
                    density[grid_x-x_left : grid_x+x_right+1,
                            grid_y-y_left : grid_y+y_right+1] += sgrid
            density = density.T        

            ## Plot contour
            fig, ax = plt.subplots(figsize=(12,10))
            plt.axis('equal')
            contour = plt.contourf(X, Y, density, 40)
            fig.colorbar(contour)
#            plt.figure(figsize=(10,10))
#            plt.axis('equal')
#            plt.contourf(X, Y, density, levels=np.arange(0, 1500, 25))
#            x_label, y_label = axis_label[x_axis], axis_label[y_axis]
            x_label, y_label = axis_label[x_axis], axis_label[y_axis]
            plt.xlabel(x_label+" axis [kpc]")
            plt.ylabel(y_label+" axis [kpc]")
            plt.xlim((-size, size))
            plt.ylim((-size, size))
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
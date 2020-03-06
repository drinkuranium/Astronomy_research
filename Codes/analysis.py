import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import sys
import time  ## For debugging purpose
from scipy.optimize import curve_fit
from pathlib import Path

#### Global varaibles
#### Code unit (Assume non-cosmological simulation)
x_unit = 1  ## In kpc
m_unit = 1  ## In 1e10 solar mass
t_unit = 1  ## In Gyr
v_unit = 1  ## In km/s
B_unit = 1  ## In Gauss


## This part of the code is from load_from_snapshot.py, written by Phil Hopkins ##

## This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO ##
def load_from_snapshot(value,ptype,sdir,snum,particle_mask=np.zeros(0),axis_mask=np.zeros(0),
        units_to_physical=False,four_char=False,snapshot_name='snapshot',snapdir_name='snapdir',extension='.hdf5'):
    '''

    The routine 'load_from_snapshot' is designed to load quantities directly from GIZMO 
    snapshots in a robust manner, independent of the detailed information actually saved 
    in the snapshot. It is able to do this because of how HDF5 works, so it  --only-- 
    works for HDF5-format snapshots [For binary-format, you need to know -exactly- the 
    datasets saved and their order in the file, which means you cannot do this, and should
    use the 'readsnap.py' routine instead.]
    
    The routine automatically handles multi-part snapshot files for you (concatenating). 
    This should work with both python2.x and python3.x

    Syntax:
      loaded_value = load_from_snapshot(value,ptype,sdir,snum,....)

      For example, to load the coordinates of gas (type=0) elements in the file
      snapshot_001.hdf5 (snum=1) located in the active directory ('.'), just call
      xyz_coordinates = load_from_snapshot('Coordinates',0,'.',1)

      More details and examples are given in the GIZMO user guide.

    Arguments:
      value: the value to extract from the HDF5 file. this is a string with the same name 
             as in the HDF5 file. if you arent sure what those values might be, setting 
             value to 'keys' will return a list of all the HDF5 keys for the chosen 
             particle type, or 'header_keys' will return all the keys in the header.
             (example: 'Time' returns the simulation time in code units (single scalar). 
                       'Coordinates' will return the [x,y,z] coordinates in an [N,3] 
                       matrix for the N resolution elements of the chosen type)
  
      ptype: element type (int) = 0[gas],1,2,3,4,5[meaning depends on simulation, see
             user guide for details]. if your chosen 'value' is in the file header, 
             this will be ignored
             
      sdir: parent directory (string) of the snapshot file or immediate snapshot 
            sub-directory if it is a multi-part file
            
      snum: number (int) of the snapshot. e.g. snapshot_001.hdf5 is '1'
            Note for multi-part files, this is just the number of the 'set', i.e. 
            if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'
      
    Optional:
      particle_mask: if set to a mask (boolean array), of length N where N is the number
        of elements of the desired ptype, will return only those elements
      
      axis_mask: if set to a mask (boolean array), return only the chosen -axis-. this
        is useful for some quantities like metallicity fields, with [N,X] dimensions 
        where X is large (lets you choose to read just one of the "X")
        
      units_to_physical: default 'True': code will auto-magically try to detect if the 
        simulation is cosmological by comparing time and redshift information in the 
        snapshot, and if so, convert units to physical. if you want default snapshot units
        set this to 'False'
        
      four_char: default numbering is that snapshots with numbers below 1000 have 
        three-digit numbers. if they were numbered with four digits (e.g. snapshot_0001), 
        set this to 'True' (default False)
        
      snapshot_name: default 'snapshot': the code will automatically try a number of 
        common snapshot and snapshot-directory prefixes. but it can't guess all of them, 
        especially if you use an unusual naming convention, e.g. naming your snapshots 
        'xyzBearsBeetsBattleStarGalactica_001.hdf5'. In that case set this to the 
        snapshot name prefix (e.g. 'xyzBearsBeetsBattleStarGalactica')
      
      snapdir_name: default 'snapdir': like 'snapshot_name', set this if you use a 
        non-standard prefix for snapshot subdirectories (directories holding multi-part
        snapshots pieces)
        
      extension: default 'hdf5': again like 'snapshot' set if you use a non-standard 
        extension (it checks multiply options like 'h5' and 'hdf5' and 'bin'). but 
        remember the file must actually be hdf5 format!

    '''



    # attempt to verify if a file with this name and directory path actually exists
    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,snapdir_name=snapdir_name,extension=extension,four_char=four_char)
    # if no valid file found, give up
    if(fname=='NULL'): 
        print('Could not find a valid file with this path/name/extension - please check these settings')
        return 0
    # check if file has the correct extension
    if(fname_ext!=extension): 
        print('File has the wrong extension, you specified ',extension,' but found ',fname_ext,' - please specify this if it is what you actually want')
        return 0
    # try to open the file
    try: 
        file = h5py.File(fname,'r') # Open hdf5 snapshot file
    except:
        print('Unexpected error: could not read hdf5 file ',fname,' . Please check the format, name, and path information is correct')
        return 0
        
    # try to parse the header
    try:
        header_toparse = file["Header"].attrs # Load header dictionary (to parse below)
    except:
        print('Was able to open the file but not the header, please check this is a valid GIZMO hdf5 file')
        file.close()
        return 0
    # check if desired value is contained in header -- if so just return it and exit
    if(value=='header_keys')|(value=='Header_Keys')|(value=='HEADER_KEYS')|(value=='headerkeys')|(value=='HeaderKeys')|(value=='HEADERKEYS')|((value=='keys' and not (ptype==0 or ptype==1 or ptype==2 or ptype==3 or ptype==4 or ptype==5))):
        q = header_toparse.keys()
        print('Returning list of keys from header, includes: ',q)
        file.close()
        return q
    if(value in header_toparse):
        q = header_toparse[value] # value contained in header, no need to go further
        file.close()
        return q

    # ok desired quantity is not in the header, so we need to go into the particle data

    # check that a valid particle type is specified
    if not (ptype==0 or ptype==1 or ptype==2 or ptype==3 or ptype==4 or ptype==5):
        print('Particle type needs to be an integer = 0,1,2,3,4,5. Returning 0')
        file.close()
        return 0
    # check that the header contains the expected data needed to parse the file
    if not ('NumFilesPerSnapshot' in header_toparse and 'NumPart_Total' in header_toparse
        and 'Time' in header_toparse and 'Redshift' in header_toparse 
        and 'HubbleParam' in header_toparse and 'NumPart_ThisFile' in header_toparse):
        print('Header appears to be missing critical information. Please check that this is a valid GIZMO hdf5 file')
        file.close()
        return 0
    # parse data needed for checking sub-files 
    numfiles = header_toparse["NumFilesPerSnapshot"]
    npartTotal = header_toparse["NumPart_Total"]
    if(npartTotal[ptype]<1): 
        print('No particles of designated type exist in this snapshot, returning 0')
        file.close()
        return 0
    # parse data needed for converting units [if necessary]
    if(units_to_physical):
        time = header_toparse["Time"]
        z = header_toparse["Redshift"]
        hubble = header_toparse["HubbleParam"]      
        cosmological = False
        ascale = 1.0;
        # attempt to guess if this is a cosmological simulation from the agreement or lack thereof between time and redshift. note at t=1,z=0, even if non-cosmological, this won't do any harm
        if(np.abs(time*(1.+z)-1.) < 1.e-6): 
            cosmological=True; ascale=time;
    # close the initial header we are parsing
    file.close()
    
    # now loop over all snapshot segments to identify and extract the relevant particle data
    check_counter = 0
    for i_file in range(numfiles):
        # augment snapshot sub-set number
        if (numfiles>1): fname = fname_base+'.'+str(i_file)+fname_ext  
        # check for existence of file
        if(os.stat(fname).st_size>0):
            # exists, now try to read it
            try: 
                file = h5py.File(fname,'r') # Open hdf5 snapshot file
            except:
                print('Unexpected error: could not read hdf5 file ',fname,' . Please check the format, name, and path information is correct, and that this file is not corrupted')
                return 0
            # read in, now attempt to parse. first check for needed information on particle number
            npart = file["Header"].attrs["NumPart_ThisFile"]
            if(npart[ptype] > 1):
                # return particle key data, if requested
                if((value=='keys')|(value=='Keys')|(value=='KEYS')): 
                    q = file['PartType'+str(ptype)].keys()
                    print('Returning list of valid keys for this particle type: ',q)
                    file.close()
                    return q
                # check if requested data actually exists as a valid keyword in the file
                if not (value in file['PartType'+str(ptype)].keys()):
                    print('The value ',value,' given does not appear to exist in the file ',fname," . Please check that you have specified a valid keyword. You can run this routine with the value 'keys' to return a list of valid value keys. Returning 0")
                    file.close()
                    return 0
                # now actually read the data
                axis_mask = np.array(axis_mask)
                if(axis_mask.size > 0):
                    q_t = np.array(file['PartType'+str(ptype)+'/'+value+'/']).take(axis_mask,axis=1)
                else:
                    q_t = np.array(file['PartType'+str(ptype)+'/'+value+'/'])                    
                # check data has non-zero size
                if(q_t.size > 0): 
                    # if this is the first time we are actually reading it, parse it and determine the shape of the vector, to build the data container
                    if(check_counter == 0): 
                        qshape=np.array(q_t.shape); qshape[0]=0; q=np.zeros(qshape); check_counter+=1;
                    # add the data to our appropriately-shaped container, now
                    try:
                        q = np.concatenate([q,q_t],axis=0)
                    except:
                        print('Could not concatenate data for ',value,' in file ',fname,' . The format appears to be inconsistent across your snapshots or with the usual GIZMO conventions. Please check this is a valid GIZMO snapshot file.')
                        file.close()
                        return 0
            file.close()
        else:
            print('Expected file ',fname,' appears to be missing. Check if your snapshot has the complete data set here')
            
    # convert units if requested by the user. note this only does a few obvious units: there are many possible values here which cannot be anticipated!
    if(units_to_physical):
        hinv=1./hubble; rconv=ascale*hinv;
        if((value=='Coordinates')|(value=='SmoothingLength')): q*=rconv; # comoving length
        if(value=='Velocities'): q *= np.sqrt(ascale); # special comoving velocity units
        if((value=='Density')|(value=='Pressure')): q *= hinv/(rconv*rconv*rconv); # density = mass/comoving length^3
        if((value=='StellarFormationTime')&(cosmological==False)): q*=hinv; # time has h^-1 in non-cosmological runs
        if((value=='Masses')|('BH_Mass' in value)|(value=='CosmicRayEnergy')|(value=='PhotonEnergy')): q*=hinv; # mass x [no-h] units

    # return final value, if we have not already
    particle_mask=np.array(particle_mask)
    if(particle_mask.size > 0): q=q.take(particle_mask,axis=0)
    return q





def check_if_filename_exists(sdir,snum,snapshot_name='snapshot',snapdir_name='snapdir',extension='.hdf5',four_char=False):
    '''
    This subroutine attempts to check if a snapshot or snapshot directory with 
    valid GIZMO outputs exists. It will check several common conventions for 
    file and directory names, and extensions. 
    
    Input:
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'
    
    Optional:
      snapshot_name: default 'snapshot': the code will automatically try a number of 
        common snapshot and snapshot-directory prefixes. but it can't guess all of them, 
        especially if you use an unusual naming convention, e.g. naming your snapshots 
        'xyzBearsBeetsBattleStarGalactica_001.hdf5'. In that case set this to the 
        snapshot name prefix (e.g. 'xyzBearsBeetsBattleStarGalactica')
      
      snapdir_name: default 'snapdir': like 'snapshot_name', set this if you use a 
        non-standard prefix for snapshot subdirectories (directories holding multi-part
        snapshots pieces)
        
      extension: default 'hdf5': again like 'snapshot' set if you use a non-standard 
        extension (it checks multiply options like 'h5' and 'hdf5' and 'bin'). but 
        remember the file must actually be hdf5 format!

      four_char: default numbering is that snapshots with numbers below 1000 have 
        three-digit numbers. if they were numbered with four digits (e.g. snapshot_0001), 
        set this to 'True' (default False)        
    '''    
    
    # loop over possible extension names to try and check for valid files
    for extension_touse in [extension,'.h5','.bin','']:
        fname=sdir+'/'+snapshot_name+'_'
        
        # begin by identifying the snapshot extension with the file number
        ext='00'+str(snum);
        if (snum>=10): ext='0'+str(snum)
        if (snum>=100): ext=str(snum)
        if (four_char==True): ext='0'+ext
        if (snum>=1000): ext=str(snum)
        fname+=ext
        fname_base=fname

        # isolate the specific path up to the snapshot name, because we will try to append several different choices below
        s0=sdir.split("/"); snapdir_specific=s0[len(s0)-1];
        if(len(snapdir_specific)<=1): snapdir_specific=s0[len(s0)-2];

        ## try several common notations for the directory/filename structure
        fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is it a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+ext; 
            fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap(snapdir)' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+snapdir_specific+'_'+ext; 
            fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is it in a snapshot sub-directory? (we assume this means multi-part files)
            fname_base=sdir+'/'+snapdir_name+'_'+ext+'/'+snapshot_name+'_'+ext; 
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is it in a snapshot sub-directory AND named 'snap' instead of 'snapshot'?
            fname_base=sdir+'/'+snapdir_name+'_'+ext+'/'+'snap_'+ext; 
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## wow, still couldn't find it... ok, i'm going to give up!
            fname_found = 'NULL'
            fname_base_found = 'NULL'
            fname_ext = 'NULL'
            continue;
        if(os.stat(fname).st_size <= 0):
            ## file exists but is null size, do not use
            fname_found = 'NULL'
            fname_base_found = 'NULL'
            fname_ext = 'NULL'
            continue;
        fname_found = fname;
        fname_base_found = fname_base;
        fname_ext = extension_touse
        break; # filename does exist! 
    return fname_found, fname_base_found, fname_ext;


##########################################################################

## This part of code is written by Dong Uk Kim, for practicing analyzing data
## Applied to GalIC test models

def load_data(value, sdir, snum, ptype_list=[0,1,2,3,4,5]):
    """This subrutine inspects the existence of halo, disk and bulge then 
    load desired date from hdf5 file if the components exist.
    
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
        element. Particle types are given in ptype_list_final.
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
            data_list[ptype] = data

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
        num_particle = load_from_snapshot('NumPart_Total', 0, sdir, snum)
        ptype_exist = np.arange(0, 6, 1, dtype=int)[np.where(num_particle>0)]
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


def plot_particles(sdir, snum, ptype_list=[0,1,2,3,4,5], sizes=[10.],
                   axes=[(0,1),(0,2),(1,2)], save=False):
    """This subruting plots the particles of type desinated by ptype_list,
    within specified size (by sizes) onto specified projected plane (by axes).
    Can save the figure to sub-directory of the data directory
    (specified by save).
    
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
    ptype_list, data_list = load_data(value, sdir, snum, ptype_list)

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

    ## Plot galaxies within given size
    axis_label = ["X", "Y", "Z"]
    for size in sizes:
        for x_axis, y_axis in axes:

            ## Plot the entire galaxy (including outermost halo particles)
            plt.figure(figsize = (10,10))
            plt.axis('equal')

            ## To prevent overwritten of one data points over other data points
            ## during plot, I plot only a fraction of data at once
            markers = ['c.', 'k.', 'r.', 'b.', 'm.', 'ko']
            markers_size = 2/size
            if markers_size < 0.1:
                markers_size = 0.1
            n = 1
            for i in range(n):
                for j in range(len(ptype_list)):
                    ptype, data = ptype_list[j], data_list[j]
                    data_len = len(data)
                    start = int(round(data_len*i/n))
                    end = int(round(data_len*(i+1)/n))
                    plt.plot(data[start:end,x_axis], data[start:end,y_axis],
                             markers[ptype], markersize=markers_size)
            x_label, y_label = axis_label[x_axis], axis_label[y_axis]
            plt.xlabel(x_label+" axis [kpc]")
            plt.ylabel(y_label+" axis [kpc]")
            plt.title("t=%.2f Gyr, projected onto %s%s-plane galaxy" %(snum*0.01, x_label, y_label))
            plt.xlim((-size, size))
            plt.ylim((-size, size))
            ## Plot center
            plt.plot(0, 0, 'ko', markersize=5)
            
#            dummy, ids = load_data('ParticleIDs', sdir, snum, ptype_list=0)
#            dummy, locs = load_data('Coordinates', sdir, snum, ptype_list=0)
#            loc = locs[0][np.where(ids[0]==205300)]
#            print(loc)
#            plt.plot(loc[0][x_axis], loc[0][y_axis], 'bo', markersize=10)
#            ## Plot the origin
            
            ## Save the figure if save flag is on
            if save:
                ## Create the directory if desniated directory does not exist
                Path(sdir+"/plot/galaxy_image/%s%s-plane"
                     %(x_label, y_label)).mkdir(parents=True, exist_ok=True)
                plt.savefig(sdir+"/plot/galaxy_image/%s%s-plane/t=%.2f Gyr particle plot.png"
                            %(x_label, y_label, snum*0.01))
            plt.show()

    return None


def plot_density_contour(sdir, snum, ptype_list=[0,1,2,3,4,5], sizes=[10.],
                   axes=[(0,1),(0,2),(1,2)], save=False):
    """This subrutine plots density contours of the particles of type desinated
    by ptype_list, within specified size (by sizes) onto specified projected
    plane (by axes). Can save the each figure to sub-directory of the data
    directory (specified by save)
    
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


def sersic_profile(r, I_e, R_e, n):
    """Compute luminosity by using Sersic profile

    Args:
      r (float): Radius

      I_e (float): Luminosity at effective radius
  
      R_e (float): Effective radius
  
      n (float): Sersic index

    Returns:
      I (float): Intensity at r according to Sersic profile
    """

    return np.log10(I_e*np.exp(-(2*n-1/3)*((r/R_e)**(1/n)-1)))


def sersic_fitting(sdir, snum, axes=[(0,1)], ptype_list=[0,1,2,3,4,5]):
    """This subrutine fits the distribution of stars to Sersic profile, and
    find out proper value of n (Sersic index) and R_e (effective radius) of
    galactic components (specified by ptype_list). Assume that the intensity
    of the stars (particles) are the same and the galaxy is spherical.
    
    Arguments:
      sdir (string): parent directory of the snapshot file or immediatei
        snapshot sub-directory if it is a multi-part file
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

    Optional:
      axes (tuple list or tuple): Specify the projected plane. Users
        can specify the projected plane by passing a list of tuple (or just 
        a tuple) of combination of 0 (for x-axis), 1 (for y-axis) or 2
        (for z-axis). For example, if one want to choose xy-plane, then pass
        axis=(0,1). Other values will give error and immediate return of this
        function. Default will inspect only xy-plane.
    
      ptype_list (int list or int): Specify particle types that will be
        plotted. Default will plot every existing components. Users can specify
        particle types by passing an array. Every particle types included in
        the array will be plotted. Just passing a int number is also okay,
        it will be treated as a list of length 1.

    Returns: Nothing
    """
    ## Load the locations of the particles
    value = 'Coordinates'
    ptype_list, data_list = load_data(value, sdir, snum, ptype_list)

    ## Treat tuple as a list of length 1
    if type(axes) == tuple: axes = [axes]

    ## Inspect legitimacy of the format
    for axis in axes:
        if axis[0] != 0 and axis[0] !=1 and axis[0] != 2 and \
        axis[1] != 0 and axis[1] !=1 and axis[1] != 2:
            print("Please follow the format: [0, 1, 2] for [x, y, z] axis - then \
                  specify the plane by pairing axis like (0,2), in case of yz-plane.")

    ptype_name = ["Gas", "Halo", "Disk", "Bulge", "Thick Disk", "BH"]
    for i in range(len(data_list)):
        for axis in axes:
            ptype, data = ptype_list[i], data_list[i], 
                
            ## Compute R_e, where half of the particles reside inside
            R = (data[:,axis[0]]**2 + data[:,axis[1]]**2)**(1/2)
            R = np.sort(R)
            R_len = len(R)
            R_e = (R[R_len//2-1]+R[R_len//2])/2 if R_len%2 else R[R_len//2]
#            if R_len%2 == 0: R_e = (R[R_len//2-1] + R[R_len//2]) / 2
#            elif R_len%2 == 1: R_e = R[R_len//2]
        
            ## Divide r domain so that each r grid contains certain number of particles
            particle_cutoff = int(R_len*0.95)  ## Exclude outliers
            resolution = 100
            particle_per_grid = int(particle_cutoff//resolution)
            r_domain_index = np.arange(0, particle_cutoff+1, particle_per_grid)
            r_domain = R[r_domain_index]
        
            ## Compute luminosity (=particle number density) as a function of R
            density = np.zeros(len(r_domain)-1)
            for j in range(len(density)):
                area = np.pi*(r_domain[j+1]**2 - r_domain[j]**2)
                density[j] = particle_per_grid/area
            luminosity = np.zeros_like(r_domain)
            for j in range(len(luminosity)-2):
                luminosity[j+1] = (density[j]+density[j+1])/2
            luminosity[0] = luminosity[1] + (luminosity[1]-luminosity[2])
            luminosity[-1] = luminosity[-2] + (luminosity[-2]-luminosity[-3])
        
            ## Compute I_e, luminosity at effective radius
            for j in range(len(r_domain)):
                if not R_e > r_domain[j]:
                    R_e_index = j
                    break
            I_e = luminosity[R_e_index]
        
            ## Find best 'n' value by using least square method
            initial_guess = [I_e, R_e, 1]
            popt, pcov = curve_fit(sersic_profile, r_domain[1:], np.log10(luminosity[1:]),
                                   p0=initial_guess, bounds=([I_e*0.98,R_e*0.981,0.1],[I_e*1.02,R_e*1.02,20]))
            ## Exclude the first element, since it incorporate large error
            print("%s - Effective radius: %.3f kpc, Sersic index: %.2f" % (ptype_name[ptype], popt[1], popt[2]))
        
            ## To find best fit in non-log scale, uncomment below line
            #popt, pcov = curve_fit(sersic_profile, r_domain[1:], luminosity[1:], p0=initial_guess, bounds=([1,R_e*0.99,0.1],[np.inf,R_e*1.01,20]))
            plt.plot(r_domain[1:], luminosity[1:], 'b-', label='Luminosity')
            plt.plot(r_domain[1:], 10**sersic_profile(r_domain[1:], *popt), 'r-', label='Sersic fit')
            plt.yscale('log')
            plt.xlabel('R (kpc)')
            plt.ylabel('Luminosity')
            plt.title("%s Sersic profile fitting" % ptype_name[ptype])
            plt.legend()
            plt.show()

    return None


def azimuthal_structure(sdir, snum):
    """This subrutine computes radial density function of the disk, then take 
    Fourier transform to the density function and plot the results.
    
    Arguments:
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

    Raises:
      If there are no disk particles, the entire program will be halted.

    Returns: Nothing
    """

    ## Load the locations of the particles
    value = 'Coordinates'
    ptype_list, data_list = load_data(value, sdir, snum, ptype_list=[2,4])

    ## Compute R for each particle
    data = data_list[0]
    data = np.vstack((data_list[0], data_list[1]))
    R = (data[:,0]**2 + data[:,1]**2)**(1/2)
    theta = np.arccos(data[:,0]/R)
    cyl = np.vstack((R, theta))
    cyl = cyl.T
    cyl = cyl[cyl[:,0].argsort()]  # Sort by R
    particle_num = len(cyl)

    ## Generate R grid and theta grid at each R
    particle_cutoff = int(particle_num*0.5)  ## Exclude outliers
    R_grid_num = 50
    R_max = cyl[particle_cutoff, 0]
    R_grid = np.linspace(0, R_max, R_grid_num+1, endpoint=True)
    R_grid_len = R_max/R_grid_num
    R_grid_point = R_grid[:-1] + R_grid_len/2
    theta_grid_num = 50
    theta_grid = np.linspace(0, 2*np.pi, theta_grid_num, endpoint=True)
    theta_grid_len = 2*np.pi/theta_grid_num
    theta_grid_point = theta_grid[:-1] + theta_grid_len/2

    R_index = 0
    fft_theta = -1  ## Just initialization
    for i in range(R_grid_num):

        ## Find particles in R_grid
        R_index_min = R_index
        while cyl[R_index, 0] < R_grid[i+1]: R_index += 1
        R_grid_data = cyl[R_index_min:R_index]

        ## Find particles in theta_grid
        R_grid_data = R_grid_data[R_grid_data[:,1].argsort()]
        particle_num_theta_grid = np.zeros_like(theta_grid_point)
        theta_index = 0
        theta_index_max = len(R_grid_data)
        for i in range(len(particle_num_theta_grid)):
            theta_index_previous = theta_index
            while theta_index < theta_index_max and R_grid_data[theta_index, 1] < theta_grid[i+1]:
                theta_index += 1
            particle_num_theta_grid[i] = theta_index - theta_index_previous

        ## Conduct FFT
        fft_theta_each_R = np.fft.fft(particle_num_theta_grid)
        normalized_fft_theta_each_R = fft_theta_each_R/fft_theta_each_R[0]
        if type(fft_theta) == int: fft_theta = normalized_fft_theta_each_R
        else: fft_theta = np.vstack((fft_theta, normalized_fft_theta_each_R))
    
    ## Artificially add center (R=0), with FFT value 0s
    R_grid_point = np.append(0, R_grid_point)
    fft_at_center = np.zeros((1, len(theta_grid_point)))
    fft_theta = np.vstack((fft_at_center, fft_theta))

    ## Plot FFT coefficient of each mode vs. R axis
    mode_max = 5
    mode_domain = np.arange(1, mode_max+1, 1)
    for mode in mode_domain:
        plt.plot(R_grid_point, abs(fft_theta[:,mode]), label="m=%i" %mode)
    time = snum*0.01
    plt.title("t=%.2f Gyr azimuthal mode analysis" %time)
    plt.xlabel("R (kpc)")
    plt.ylabel("a_m/a_0")
    plt.legend()
    plt.show()

    return None


def measure_SFR(sdir, snum_max):
    """Measure the star formation rate (SFR) of the entire galaxy by plotting
    the number of gas and newly created star particles.

    Arguments:
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file
            
      snum_max (int): Maximum number of the snapshot.

    Raises:
      If there are no disk particles, the entire program will be halted.

    Returns: None
    """

    ## Read the number of ptype0 and ptype4 particles of each snapshot
    snum_list = np.arange(0, snum_max+1, 1, dtype=int)
    t_list = np.array([])
    num_ptype0_list, num_ptype4_list = np.zeros(snum_max+1), np.zeros(snum_max+1)
    SFR_list = np.zeros(snum_max+1)
    for snum in snum_list:
        num_ptypes = load_from_snapshot('NumPart_Total', 0, sdir, snum)
        num_ptype0_list[snum] = num_ptypes[0]
        num_ptype4_list[snum] = num_ptypes[4]
        SFR_list[snum] = np.sum(load_from_snapshot('StarFormationRate', 0, sdir, snum))
        t = load_from_snapshot('Time', 0, sdir, snum)*t_unit
        t_list = np.append(t_list, t)

    ## Plot the number of particles and SFR
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (Gyr)')
    ax1.set_ylabel('Number of particles')
    ax1.plot(t_list, num_ptype0_list, 'c-', label='ptype0')
    ax1.plot(t_list, num_ptype4_list, 'm-', label='ptype4')
    plt.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel('SFR (M_sun/yr)')
    ax2.plot(t_list, SFR_list, 'r-', label='SFR')
    fig.tight_layout()
    plt.title('Snapshot number vs ptype0 and ptype4 number')
    plt.legend()
    plt.show()

    return None


def measure_mass_within_radius_2D(sdir, snum, radius, axis, ptype_list=[0,1,2,3,4,5]):
    """This function measures the mass of galactic components projected onto
    specified plane (by axis) within 'radius'. 

    Arguments:
      sdir (string): parent directory of the snapshot file or immediate 
        snapshot sub-directory if it is a multi-part file

      snum (int): number (int) of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

      radius (int or float): Radius that user wants to measure the mass within.

      axis (tuple): Specify the projected plane. Users can specify the
        projected plane by passing a tuple (not list in this function)
        of combination of 0 (for x-axis), 1 (for y-axis) or 2 (for z-axis).
        For example, if one want to choose xy-plane, then pass axis=(0,1).
        Other values will give error and immediate return of this function.

    Optional:
      ptype_list (int list or int): Specify particle types that will be
        plotted. Default will plot every existing components. Users can specify
        particle types by passing an array. Every particle types included in
        the array will be plotted. Just passing a int number is also okay,
        it will be treated as a list of length 1.

    Returns:
      M_r (float): The mass within the sphere of 'radius'
    """

    ## Load the locations of the particles
    value = 'Coordinates'
    ptype_list, data_list = load_data(value, sdir, snum, ptype_list)

    mass_each_ptype = np.zeros_like(ptype_list)
    for data in data_list:
        a = 0

    M_r = 0
    return M_r


def plot_mass_vs_radius_2D(sdir, snum, radius, axes=[(0,1),(0,2),(1,2)],
                           ptype_list=[0,1,2,3,4,5]):
    """This function plots the mass of galactic components projected onto
    specified plane (by axis) within 'radius' versus radius. 

    Arguments:
      sdir (string): parent directory of the snapshot file or immediate 
        snapshot sub-directory if it is a multi-part file

      snum (int): number (int) of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

      radius (int or float): Radius that user wants to plot the mass within.

    Optional:
      axes (tuple list or tuple): Specify the projected plane. Users
        can specify the projected plane by passing a list of tuple (or just 
        a tuple) of combination of 0 (for x-axis), 1 (for y-axis) or 2
        (for z-axis). For example, if one want to choose xy-plane, then pass
        axis=(0,1). Other values will give error and immediate return of this
        function. Default will inspect only xy-plane.

    Returns:
      M_r (float): The mass within the sphere of 'radius'
    """

    ## Treat tuple as a list of length 1
    if type(axes) == tuple: axes = [axes]

    ## Inspect legitimacy of the format
    for axis in axes:
        if axis[0] != 0 and axis[0] !=1 and axis[0] != 2 and \
        axis[1] != 0 and axis[1] !=1 and axis[1] != 2:
            print("Please follow the format: [0, 1, 2] for [x, y, z] axis - then \
                  specify the plane by pairing axis like (0,2), in case of yz-plane.")

## Select model
sdir = "/home/du/gizmo/TestGas/Test_Gas_Generation/results"
snum = 15

## Plot the locations of the particles
plot_particles(sdir, snum, sizes=[10], axes=(0,1), ptype_list=[0,2,4], save=True)

## Plot the density contours
a = time.time()
plot_density_contour(sdir, snum, sizes=[10], axes=(0,1), ptype_list=[0,2,4], save=True)
print(time.time()-a)
## Fit the luminosity curve to Sersic profile
#sersic_fitting(sdir, snum, axes=(0,1))

## Conduct Fourier analysis of the density of the disk to investigate bar and spiral structures
#azimuthal_structure(sdir, snum)

## Investigate SFR
#max_snum = 100
#measure_SFR(sdir, max_snum)

## Plot the evolution consequnce of the galaxy and save them all
#snum_list = np.arange(0, 101, 1)
#for snum in snum_list:
#    plot_particles(sdir, snum, sizes=10, axes=[(0,1)], save=False, ptype_list=[2,0,4])
#    plot_density_contour(sdir, snum, sizes=[10], axes=(0,1), ptype_list=[1,2,4,0], save=False)

#dummy, loc = load_data('Coordinates', sdir, snum, [0,2])
#dummy, vel = load_data('Velocities', sdir, snum, [0,2])
#R0 = np.sqrt(loc[0][:,0]**2+loc[0][:,1]**2)
#R1 = np.sqrt(loc[1][:,0]**2+loc[1][:,1]**2)
#v0 = np.sqrt(vel[0][:,0]**2+vel[0][:,1]**2)
#v1 = np.sqrt(vel[1][:,0]**2+vel[1][:,1]**2)
#v0 = v0[np.argsort(R0)]
#R0 = np.sort(R0)
#
#plt.plot(R1, v1, 'b+')
#plt.plot(R0, v0, 'r-')
#plt.show()
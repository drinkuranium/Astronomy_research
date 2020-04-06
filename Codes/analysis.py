import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
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
            if(npart[ptype] >= 1):
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

def load_data(value, ptype_list, sdir, snum, find_center=True):
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

      ptype_list (int list or int): Specify particle types that will be loaded.
             
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

    Optionals:
      find_center (boolean): Choose whether find the exact location of
        the center by finding density maximum or not. Default is True.
        Note that it involves 3D kernel density estimation which is quite slow.
        So you can disable this if you don't need exact location of the center.

    Returns:
      ptype_exist (int list): Store ptypes which exist in the data.

      data_list (object ndarray): Store data corresponds to the 'value' in each
        element. Particle types are given in ptype_list_final.
    """
    
    ## Make int ptype into array
    if type(ptype_list) == int: ptype_list = [ptype_list]

    ## Load data of desinated particle types
    data_list = np.array([0,0,0,0,0,0], dtype=object)
    ptype_exist = ptype_list[:]
    for ptype in ptype_list:
        data = load_from_snapshot(value, ptype, sdir, snum)
        if type(data) == int and data == 0:  ## When particles do not exist
            print("ptype %i particle dose not exist in the galaxy." %ptype)
            ptype_exist.remove(ptype)
        else:
            data_list[ptype] = data

    ## Immediately exit program if there are no particles of desniated ptypes
    if len(ptype_exist) == 0:
        sys.exit("Particles of desinated ptypes do not exist at all. Change ptypes.")
    
    ## Select data of only existing particles
    data_list = data_list[ptype_exist]

    ## Make correction to postion and velocity data so that
    ## Position: the galactic center (center of mass) is located at (0,0).
    ## Velocity: the net momentum becomes 0.
    if value == 'Coordinates' or value == 'Velocities':
        m_total = 0
        cm = 0
        num_particle = load_from_snapshot('NumPart_Total', 0, sdir, snum)
        ptype_exist_temp = np.arange(0, 6, 1, dtype=int)[np.where(num_particle>0)]
        for ptype in ptype_exist_temp:
            ## Negelect halo particle, they might affect CM large while
            ## merely affect on the galactic center
            if ptype == 1:
                continue
            m = load_from_snapshot('Masses', ptype, sdir, snum)
            m_total += sum(m)
            temp_data = load_from_snapshot(value, ptype, sdir, snum)
            cm += np.dot(m, temp_data)
        cm /= m_total
        for data in data_list:
            data -= cm

    ## Find galactic center by finding density maxima when find_center is True
    opt=False
    if find_center and value == 'Coordinates':
        ## Load the locations of the particles
        value = 'Coordinates'
        ptype_list_temp = [0,2,3,4]
        ptype_exist_temp = []
        num_particle = load_from_snapshot('NumPart_Total', 0, sdir, snum)
        for ptype in np.arange(0, 6, 1, dtype=int)[np.where(num_particle>0)]:
            if ptype in ptype_list_temp:
                ptype_exist_temp.append(ptype)
        data_list_temp = np.zeros_like(ptype_exist_temp, dtype=object)
        for i in range(len(ptype_exist_temp)):
            ptype = ptype_exist_temp[i]
            data_list_temp[i] = load_from_snapshot(value, ptype, sdir, snum)
            data_list_temp[i] -= cm

        ## Load mass data
        value = 'Masses'
        masses = np.zeros_like(ptype_exist_temp, dtype=np.float32)
        for i in range(len(ptype_exist_temp)):
            ptype = ptype_exist_temp[i]
            mass = load_from_snapshot(value, ptype, sdir, snum)
            masses[i] = mass[0]  ## Assume mass of each ptype particle is the same

        ## Assume the galactic center is within size, set values
        grid_num, size = 41, 3
        grid_len = 2*size/(grid_num-1)

        ## Using float32 instead of float64 results in maximum ~1e-5 of
        ## fractional error in the final result.
        ## For strict analysis use float64, but nomrally float32 is enough.
        my_dtype = np.float32
        density = np.zeros((grid_num, grid_num, grid_num), dtype=my_dtype)
        
        ## Remove data out of the contour domain 
        for i in range(len(ptype_exist_temp)):
            ptype, data, mass = ptype_exist_temp[i], data_list_temp[i], masses[i]
            ## Compute bandwidth
            if ptype == 4 and len(data) < 100:
                ## Sometimes SFR is so low, so number of ptype4 can be
                ## very small. In this case, use gas data together.
                gas_data = load_from_snapshot('Coordinates', 0, sdir, snum)
                gas_data = np.vstack((gas_data, data))
                sigma = np.std(gas_data, axis=0)
                h = sigma*(4/5/len(gas_data))**(1/7)  ## Silverman's rule in 3D
            else:
                sigma = np.std(data, axis=0)
                h = sigma*(4/5/len(data))**(1/7)  ## Silverman's rule in 3D
            ## Arbitrarily decrease bandwidth
            ## To avoid over-smoothing central dense region
            resolution = 1  ## Arbitrary number! - Rule of thumb is 2
            h /= resolution

            ## Regulate memory by reducing kernel size if it is too big
            origin = (2*h/grid_len).astype(int)
            memory = 9*(origin[0]+1)*(origin[1]+1)*(origin[2]+1)*(sys.getsizeof(my_dtype())-24) ## Predicted maximum allocation memory
            maximum_memory = 24*1024**3  ## Allocation memory limit
            if memory > maximum_memory:
                reduce_factor = (maximum_memory/memory)**(1/2)
                h *= reduce_factor
                origin = (2*h/grid_len).astype(int)

            ## Construct matrix that stores the value of kernel function
            ## Kernel function: Cubic spline
            ## I try to use numpy module as much as possible for performance
            ## First construct 3 matrices which stores the value of cubic
            ## spline kernel in each domain
            ## In this code, kernel_temp0: for 0<=q<1
            ##               kernel_temp1: for 1<=q<2
            ##               kernel_temp2: for 2<=q
            ## Then pile up them into 3D array and sort each element, then
            ## pick medium value - then it becomes cubic spline kernel.
            ## First, construct only an octant of kernel matrix
#            time_measure1 = time.time()
            x = np.zeros(origin+1, dtype=my_dtype)
            for j in range(len(x)):
                x[j] = j
            y = np.zeros(origin+1, dtype=my_dtype)
            for j in range(len(y[0])):
                y[:,j] = j
            z = np.zeros(origin+1, dtype=my_dtype)
            for j in range(len(z[0,0])):
                z[:,:,j] = j
            q = np.zeros(1, dtype=my_dtype)  ## Declear dtype first for memory
            q = grid_len*((x/h[0])**2 + (y/h[1])**2 + (z/h[2])**2)**(1/2)
            del(x, y, z)  ## Some objects require too big memory, so free them
            kernel_temp = np.zeros((np.append(3, origin+1)), dtype=my_dtype)
            kernel_temp[1] = (2-q)**3
            ## Faster than just 4-6*kernel_temp[0]**2+3*kernel_temp[0]**3
            kernel_temp[0] = -3*kernel_temp[1]+12*(q-1.5)**2+1
            del(q)
            kernel_temp = np.sort((kernel_temp), axis=0)
            kernel_quarter = np.copy(kernel_temp[1])
            del(kernel_temp)
            ## Construct full kernel matrix using symmetry to origin
            kernel = np.zeros(origin*2+1, dtype=my_dtype)
            kernel[origin[0]:, origin[1]:, origin[2]:] = kernel_quarter
            kernel[:origin[0], origin[1]:, origin[2]:] = np.flip(kernel_quarter[1:,:,:], 0)
            kernel[origin[0]:, :origin[1], origin[2]:] = np.flip(kernel_quarter[:,1:,:], 1)
            kernel[:origin[0], :origin[1], origin[2]:] = np.flip(kernel_quarter[1:,1:,:], (0,1))
            kernel[origin[0]:, origin[1]:, :origin[2]] = np.flip(kernel_quarter[:,:,1:], 0)
            kernel[:origin[0], origin[1]:, :origin[2]] = np.flip(kernel_quarter[1:,:,1:], (0,2))
            kernel[origin[0]:, :origin[1], :origin[2]] = np.flip(kernel_quarter[:,1:,1:], (1,2))
            kernel[:origin[0], :origin[1], :origin[2]] = np.flip(kernel_quarter[1:,1:,1:], (0,1,2))
            kernel /= np.sum(kernel)  ## Normalization for conservation
            kernel *= mass/grid_len**2  ## Assign density
#            time_measure2 = time.time()
#            print("Kernel shape: ", np.shape(kernel))
#            print("Kernel construction time:", time_measure2-time_measure1)

            ## Since kernel function applied, even particles which are not
            ## in the target region can affect density. So I need to
            ## include that particles - what extra means
            extra = (2*h/grid_len).astype(int)*grid_len

            index = np.where((abs(data[:,0])<size+extra[0])
                             & (abs(data[:,1])<size+extra[1])
                             & (abs(data[:,2])<size+extra[2]))
            data = data[index]
            ## Assign particles to grid points and construct density
            data += size + size/grid_num/2
            grid_index = (np.floor(data/grid_len)).astype(int)
            if opt:
                particle_mesh = np.zeros((grid_num+2*origin))
                for i in range(len(grid_index)):
                    particle_mesh[grid_index[i]+origin] += 1
                nonzero_mesh = np.where(particle_mesh > 0)
                grid_index = nonzero_mesh-origin
            origin_array = origin*np.ones_like(grid_index)
            ## Consider boundaries and set limit of kernel approapriately
            x_left = np.min((grid_index[:,0], origin_array[:,0]), axis=0)
            x_right = np.min((grid_num-grid_index[:,0]-1, origin_array[:,0]), axis=0)
            y_left = np.min((grid_index[:,1], origin_array[:,1]), axis=0)
            y_right = np.min((grid_num-grid_index[:,1]-1, origin_array[:,1]), axis=0)
            z_left = np.min((grid_index[:,2], origin_array[:,2]), axis=0)
            z_right = np.min((grid_num-grid_index[:,2]-1, origin_array[:,2]), axis=0)
            for j in range(len(grid_index)):
                sgrid = kernel[origin[0]-x_left[j] : origin[0]+x_right[j]+1,
                               origin[1]-y_left[j] : origin[1]+y_right[j]+1,
                               origin[2]-z_left[j] : origin[2]+z_right[j]+1]
                ## For opt=True, number of particle in each grid must be considered.
                if opt:
                    pnum = particle_mesh[grid_index[j]+origin]
                    density[grid_index[j,0]-x_left[j] : grid_index[j,0]+x_right[j]+1,
                            grid_index[j,1]-y_left[j] : grid_index[j,1]+y_right[j]+1,
                            grid_index[j,2]-z_left[j] : grid_index[j,2]+z_right[j]+1] += sgrid*pnum
                else:
                    density[grid_index[j,0]-x_left[j] : grid_index[j,0]+x_right[j]+1,
                            grid_index[j,1]-y_left[j] : grid_index[j,1]+y_right[j]+1,
                            grid_index[j,2]-z_left[j] : grid_index[j,2]+z_right[j]+1] += sgrid
            del(sgrid, kernel, kernel_quarter)
#            print("Particle assign time:", time.time()-time_measure2,
#                  "(particle number: %d)\n" % len(data))
        
        ## Parabolic interpolation to find density maximum location
        max_ind = np.unravel_index(np.argmax(density), density.shape)
        neighbor1 = np.array([density[tuple(max_ind-np.array([1,0,0]))],
                              density[tuple(max_ind-np.array([0,1,0]))],
                              density[tuple(max_ind-np.array([0,0,1]))]])
        neighbor2 = np.array([density[tuple(max_ind+np.array([1,0,0]))],
                              density[tuple(max_ind+np.array([0,1,0]))],
                              density[tuple(max_ind+np.array([0,0,1]))]])
        ind_correction = (neighbor1-neighbor2)/2/(neighbor1-2*density[max_ind]+neighbor2)
        max_ind += ind_correction
        
#        ## Debugging purpose
#        print(max_ind*grid_len-size)
#        x_grid = np.linspace(-size, size, grid_num, endpoint=True)
#        y_grid = np.linspace(-size, size, grid_num, endpoint=True)
#        X, Y = np.meshgrid(x_grid, y_grid)
#        plt.contourf(X, Y, np.sum(density.T, axis=0))
#        plt.show()

        ## Shift coordinate
        for data in data_list:
            data -= (max_ind*grid_len - size)

    return ptype_exist, data_list


def kernel_density_estimation_2D(sdir, snum, ptype_list, size, plane, grid_num,
                                 resolution=2., opt=False, debug=False):
    """This subrutine create density matrix by using cubic spline kernel.
    
    Arguments:
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file.
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

      ptype_list (int list or int): Specify particle types that will be
        plotted.

      size (float): The size of the region that the estimation will be done
        in code unit (nomarlly kpc - but inspect your GIZMO setting!).
        Special value -1 - then the size of the plot will be scaled to
        incorporate every particles.
    
      plane (tuple): Specify the projected plane. You can specify the projected
        plane by passing a tuple of combination of 0 (for x-axis),
        1 (for y-axis) or 2 (for z-axis). For example, if you want to choose
        xy-plane, then pass (0,1).

      grid_num (int): The number of grid points of the resulting contour.

    Optionals:
      resolution (float): Arbitrarily decrease bandwidth of the kernel
        to avoid over-smoothing of central dense region. Default is 2.
        Increasing it will give inaccurate results in sparse regions, and
        increasing it too much will give inaccurate results even at the center.

      opt (boolean): Decide whether execute different algorithm or not. 
        This is faster when number density is high. Recommendeed for large
        particle number or small grid number for.
        Rule of thumb: If average number of particles in each grid point > 0.6,
        then turn on this. But keep in mind that this will be vary depeding on
        the distribution of the particles. Default is False.

      debug (boolean): Turn on this when you need to check performance or
        memory usage. Default is False.

    Returns:
      density (2D ndarray): The matrix that stores density
    """

    ## Load the locations and masses of the particles
    ptype_exist, data_list = load_data('Coordinates', ptype_list, sdir, snum)
    masses = np.zeros_like(ptype_exist, dtype=float)
    for i in range(len(ptype_exist)):
        ptype = ptype_exist[i]
        mass = load_from_snapshot('Masses', ptype, sdir, snum)
        masses[i] = mass[0]  ## Assume mass of each ptype particle is the same

    ## Using float32 instead of float64 results in maximum ~1e-5 of
    ## fractional error in the final result.
    ## For strict analysis use float64, but nomrally float32 is enough.
    my_dtype = np.float32
    ## Density of each grid
    density = np.zeros((grid_num, grid_num), dtype=my_dtype)
    grid_len = 2*size/grid_num
    x_axis, y_axis = plane
    
    ## Remove data out of the contour domain
    for i in range(len(ptype_exist)):
        ptype, data = ptype_exist[i], data_list[i]
        if debug: print("ptype%d" %ptype)
        ## Compute bandwidth
        if ptype == 4 and len(data) < 1000:
            ## Sometimes SFR is so low, so number of ptype4 can be
            ## very small. In this case, use gas data together.
            dummy, gas_data = load_data('Coordinates', 0, sdir, snum)
            gas_data = np.vstack((gas_data[0], data))
            sigma = np.std(gas_data[:,[x_axis,y_axis]], axis=0)
            h = sigma*len(gas_data)**(-1/6)  ## Silverman's rule in 2D
        else:
            sigma = np.std(data[:,[x_axis,y_axis]], axis=0)
            h = sigma*len(data)**(-1/6)  ## Silverman's rule in 2D
        ## Arbitrarily decrease bandwidth
        ## To avoid over-smoothing central dense region
        h1, h2 = h[0]/resolution, h[1]/resolution

        ## Construct matrix that stores the value of kernel function
        ## Kernel function: Cubic spline
        origin_x = int(2*h1/grid_len)
        origin_y = int(2*h2/grid_len)
        ## Regulate memory by reducing kernel size if it is too big
        memory = 5*(origin_x+1)*(origin_y+1)*(sys.getsizeof(my_dtype())-24) ## Predicted maximum allocation memory
        maximum_memory = 24*1024**3  ## Allocation memory limit
        if memory > maximum_memory:
            reduce_factor = (maximum_memory/memory)**(1/2)
            h1 *= reduce_factor
            h2 *= reduce_factor
            origin_x = int(2*h1/grid_len)
            origin_y = int(2*h2/grid_len)
            print("Program requires too much memory. Kernel is too large.")
            print("Reducing kernel size by %.4f..." % reduce_factor)
            print("Results can be inaccurate.")
            print("To avoid this, reduce grid number or expand plot area.")
        ## I try to use numpy module as much as possible for performance
        ## First construct 3 matrices which stores the value of cubic
        ## spline kernel in each domain
        ## In this code, kernel_temp0: for 0<=q<1
        ##               kernel_temp1: for 1<=q<2
        ##               kernel_temp2: for 2<=q
        ## Then pile up them into 3D array and sort each element, then
        ## pick medium value - then it becomes cubic spline kernel.
        if debug: time_measure1 = time.time() 
        ## First, construct only a quarter of kernel matrix
        x = np.outer(np.arange(0, origin_x+1, 1, dtype=my_dtype),
                     np.ones(origin_y+1, dtype=my_dtype))
        y = np.outer(np.ones(origin_x+1, dtype=my_dtype),
                     np.arange(0, origin_y+1, 1, dtype=my_dtype))
        q = np.zeros(1, dtype=my_dtype)  ## Declear dtype first for memory
        q = grid_len*((x/h1)**2 + (y/h2)**2)**(1/2)
        del(x, y)  ## Some objects require too big memory, so free them
        kernel_temp = np.zeros((3, origin_x+1, origin_y+1), dtype=my_dtype)
        kernel_temp[1] = (2-q)**3
        ## Faster than just 4-6*kernel_temp[0]**2+3*kernel_temp[0]**3
        kernel_temp[0] = -3*kernel_temp[1]+12*(q-1.5)**2+1
        del(q)
        kernel_temp = np.sort((kernel_temp), axis=0)
        kernel_quarter = np.copy(kernel_temp[1])
        del(kernel_temp)
        ## Construct full kernel matrix using symmetry to origin
        kernel = np.zeros((origin_x*2+1, origin_y*2+1), dtype=my_dtype)
        kernel[origin_x:, origin_y:] = kernel_quarter
        kernel[:origin_x+1, :origin_y+1] = np.flip(kernel_quarter, (0,1))
        kernel[origin_x+1:, :origin_y] = np.flip(kernel_quarter[1:,1:], 1)
        kernel[:origin_x, origin_y+1:] = np.flip(kernel_quarter[1:,1:], 0)
        kernel /= np.sum(kernel)  ## Normalization for conservation
        kernel *= masses[i]/grid_len**2  ## Assign density
        time_measure2 = time.time()
        if debug:
            print("Kernel matrix shape:", np.shape(kernel))
            print("Kernel construction time:", time_measure2-time_measure1)

        ## Since kernel function applied, even particles which are not
        ## in the target region can affect density. So I need to
        ## include that particles - what extra means
        extra_x = int(2*h1/grid_len)*grid_len
        extra_y = int(2*h2/grid_len)*grid_len
        index = np.where((abs(data[:,x_axis])<size+extra_x)
                         & (abs(data[:,y_axis])<size+extra_y))
        data = data[index]
        ## Assign particles to grid points and construct density
        data += size + size/grid_num/2
        grid_x = (np.floor(data[:,x_axis]/grid_len)).astype(int)
        grid_y = (np.floor(data[:,y_axis]/grid_len)).astype(int)

        ## When you choose opt=True, assign particles to the mesh
        ## then multiply kernels to each grid point of the mesh
        if opt:
            particle_mesh = np.zeros((grid_num+2*origin_x, grid_num+2*origin_y))
            for i in range(len(grid_x)):
                particle_mesh[grid_x[i]+origin_x, grid_y[i]+origin_y] += 1
            nonzero_mesh = np.where(particle_mesh > 0)
            grid_x, grid_y = nonzero_mesh[0]-origin_x, nonzero_mesh[1]-origin_y
        origin_x_array = origin_x*np.ones_like(grid_x)
        origin_y_array = origin_y*np.ones_like(grid_y)
        ## Consider boundaries and set limit of kernel approapriately
        x_left = np.min((grid_x, origin_x_array), axis=0)
        x_right = np.min((grid_num-grid_x-1, origin_x_array), axis=0)
        y_left = np.min((grid_y, origin_y_array), axis=0)
        y_right = np.min((grid_num-grid_y-1, origin_y_array), axis=0)
        for j in range(len(grid_x)):
            sgrid = kernel[origin_x-x_left[j] : origin_x+x_right[j]+1,
                           origin_y-y_left[j] : origin_y+y_right[j]+1]
            ## For opt=True, number of particle in each grid must be considered.
            if opt:
                pnum = particle_mesh[grid_x[j]+origin_x, grid_y[j]+origin_y]
                density[grid_x[j]-x_left[j] : grid_x[j]+x_right[j]+1,
                        grid_y[j]-y_left[j] : grid_y[j]+y_right[j]+1] += sgrid*pnum
            else:
                density[grid_x[j]-x_left[j] : grid_x[j]+x_right[j]+1,
                        grid_y[j]-y_left[j] : grid_y[j]+y_right[j]+1] += sgrid
        del(sgrid, kernel, kernel_quarter)
        if debug:
            print("Particle assign time:", time.time()-time_measure2,
                  "(particle number: %d)\n" % len(data))

    if debug:
        x_grid = np.linspace(-size, size, grid_num, endpoint=True)
        y_grid = np.linspace(-size, size, grid_num, endpoint=True)
        X, Y = np.meshgrid(x_grid, y_grid)
        plt.contourf(X, Y, np.sum(np.flip(density.T, axis=0), axis=0))
        plt.show()

    return density



def plot_particles(sdir, snum, ptype_list, size, plane, trackID=0, find_center=True, save=False):
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

      ptype_list (int list or int): Specify particle types that will be
        plotted.

      size (float): The size of the plot in code unit
        (nomarlly kpc - but inspect your GIZMO setting!).
        Special value -1 - then the size of the plot will be scaled to
        incorporate every particles.
    
      plane (tuple): Specify the projected plane. You can specify the projected
        plane by passing a tuple of combination of 0 (for x-axis),
        1 (for y-axis) or 2 (for z-axis). For example, if you want to choose
        xy-plane, then pass (0,1).

    Optionals:
      trackID (int): Choose a particle by its ID that you want to track.
        Default is 0, in this case, does not track any particles.

      find_center (boolean): Choose whether find the exact location of
        the center by finding density maximum or not. Default is True.
        Note that it involves 3D kernel density estimation which is quite slow.
        So you can disable this if you don't need exact location of the center.

      save (boolean): Determine whether save the plot or not. Default is False.
        You can specify the saving directory. Default will be a sub-directory
        of the data directory, and if the directory does not exist,
        then it will create the directory.

    Returns: Nothing
    """

    ## Load the locations of the particles
    value = 'Coordinates'
    ptype_exist, data_list = load_data(value, ptype_list, sdir, snum, find_center=find_center)

    ## Plot the entire galaxy (including outermost halo particles)
    plt.rc('font', size=15)

    plt.figure(figsize = (10,10))
    plt.axis('equal')
   
    ## To prevent overwritten of one data points over other data points
    ## during plot, I plot only a fraction of data at once
    markers = ['c.', 'k.', 'r.', 'b.', 'm.', 'ko']
    markers_size = 10/size
    if markers_size < 0.1:
        markers_size = 0.1
    ## Increase n to avoid overwrapping of plotted particles
    x_axis, y_axis = plane
    n = 1
    for i in range(n):
        for j in range(len(ptype_exist)):
            ptype, data = ptype_exist[j], data_list[j]
            data_len = len(data)
            start = int(round(data_len*i/n))
            end = int(round(data_len*(i+1)/n))
            plt.plot(data[start:end,x_axis], data[start:end,y_axis],
                     markers[ptype], markersize=markers_size)
    axis_label = ["X", "Y", "Z"]
    x_label, y_label = axis_label[x_axis], axis_label[y_axis]
    plt.xlabel(x_label+" axis [kpc]")
    plt.ylabel(y_label+" axis [kpc]")
    plt.title("t=%.2f Gyr, projected onto %s%s-plane, only gas particles" %(snum*0.01, x_label, y_label))
    plt.xlim((-size, size))
    plt.ylim((-size, size))
    ## Plot the origin
    plt.plot(0, 0, 'ko', markersize=5)

    ## Emphasize the particle that you want to track
    if trackID != 0:
        ## Find the particle
        dummy, ids = load_data('ParticleIDs', ptype_exist, sdir, snum)
        dummy, locs = load_data('Coordinates', ptype_exist, sdir, snum, find_center=find_center)
        for ptype_index in range(len(ptype_exist)):
            part_index = np.where(ids[ptype_index]==trackID)
            if len(part_index[0]) >= 1:
                break
        ## Plot the particle when it exists
        if len(part_index[0]) < 1:
            print("Particle does not exist. Please enter valid particle ID.")
        elif len(part_index[0]) == 1:
            track_markers = ['c*', 'k*', 'r*', 'b*', 'm*', 'k*']
            loc = locs[ptype_index][part_index[0]][0]
            ptype_trackID = ptype_exist[ptype_index]
            ## When the particle is out of the plot
            if abs(loc[x_axis]) > size or abs(loc[y_axis]) > size:
                print("Particle %d is out of the plot." %trackID)
            plt.plot(loc[x_axis], loc[y_axis],
                     track_markers[ptype_trackID], markersize=15)
        else:
            ## Should not reach here
            sys.exit("Error: More than 1 particles have the same ID")
  
    ## Save the figure if save flag is on
    if save:
        ## Create the directory if desniated directory does not exist
        Path(sdir+"/plot/galaxy_image/%s%s-plane"
             %(x_label, y_label)).mkdir(parents=True, exist_ok=True)
        plt.savefig(sdir+"/plot/galaxy_image/%s%s-plane/t=%.2f Gyr particle plot.png"
                    %(x_label, y_label, snum*0.01))
    plt.show()
    plt.rc('font', size=10)

    return None


def plot_density_contour(sdir, snum, ptype_list, size, plane,
                         grid_num=400, save=False, opt=False):
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

      ptype_list (int list or int): Specify particle types that will be
        plotted.

      size (float): The size of the plot in code unit
        (nomarlly kpc - but inspect your GIZMO setting!).
        Special value -1 - then the size of the plot will be scaled to
        incorporate every particles.
    
      plane (tuple): Specify the projected plane. You can specify the projected
        plane by passing a tuple of combination of 0 (for x-axis),
        1 (for y-axis) or 2 (for z-axis). For example, if you want to choose
        xy-plane, then pass (0,1).

    Optionals:
      grid_num (int): The number of grid points of the resulting contour.
        Default is 400, and increasign further hardly changes the results.

      save (boolean): Determine whether save the plot or not. Default is False.
        You can specify the saving directory. Default will be a sub-directory
        of the data directory, and if the directory does not exist,
        then it will create the directory.

      opt (boolean): Decide whether execute different algorithm or not. 
        This is faster when number density is high. Recommendeed for large
        particle number or small grid number for.
        Rule of thumb: If average number of particles in each grid point > 0.6,
        then turn on this. But keep in mind that this will be vary depeding on
        the distribution of the particles. Default is false.

    Returns: None
    """

    grid_num += 1  ## For inner process, resulting grid number is still the same

    ## Create meshgrid to plot contour
    x_grid = np.linspace(-size, size, grid_num, endpoint=True)
    y_grid = np.linspace(-size, size, grid_num, endpoint=True)
    X, Y = np.meshgrid(x_grid, y_grid)

    density = kernel_density_estimation_2D(sdir, snum, ptype_list, size, plane,
                                           grid_num, opt=opt, debug=False)

    ## Plot contour
    fig, ax = plt.subplots(figsize=(12,9.9))
    plt.axis('equal')
    contour = plt.contourf(X, Y, np.flip(density.T, axis=0), 40)
    fig.colorbar(contour)
#    plt.figure(figsize=(10,10))
#    plt.axis('equal')
#    plt.contourf(X, Y, np.flip(density.T, axis=0), levels=np.arange(0, 0.01, 0.0005))
    axis_label = ["X", "Y", "Z"]
    x_axis, y_axis = plane
    x_label, y_label = axis_label[x_axis], axis_label[y_axis]
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


def sersic_fitting(sdir, snum, plane, kernel=True):
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

      plane (tuple): Specify the projected plane. You can specify the projected
        plane by passing a tuple of combination of 0 (for x-axis),
        1 (for y-axis) or 2 (for z-axis). For example, if you want to choose
        xy-plane, then pass (0,1).

    Optionals:
      kernel (boolean): Choose whether use kernel density estimation or not.
        Using kernel density estimation is more accurate, but slower.

    Returns: Nothing
    """
    
    if kernel:
        ## Set grid for fitting
        ptype_exist, data_list = load_data('Coordinates', [1,2,3,4], sdir, snum)
        
        ## Sersic fitting for halo and bulge components
        ptype_name = ["Halo", "Bulge"]
        for i in range(len(ptype_name)):
            ptype = 2*i+1
            ## Continue to next ptype if ptype does not exist
            if ptype not in ptype_exist:
                break
            ptype_index = np.where(np.array(ptype_exist) == ptype)[0][0]
            data = data_list[ptype_index]
            
            ## Construct R_grid 
            x_axis, y_axis = plane
            R = (data[:,x_axis]**2 + data[:,y_axis]**2)**(1/2)
            R = np.sort(R)
            R_len = len(R)
            particle_cutoff = int(R_len*0.95)
            R_max = R[particle_cutoff] 
            R_domain = np.linspace(0, R_max, 101, endpoint=True)

            ## Effective radius can be computed directly by sorting R
            R_e = (R[R_len//2-1]+R[R_len//2])/2 if R_len%2 else R[R_len//2]

            density_grid_len = 0.02
            ## To deal with boundary (R==size) grid, include more than specified size
            density_grid_num = 2*round(size/density_grid_len)+2
            density = kernel_density_estimation_2D(sdir, snum, ptype, R_max,
                                                   (0,1), density_grid_num, resolution=5, opt=False)
            x = np.outer(np.arange(-size+density_grid_len/2, R_max, density_grid_len, dtype=float),
                     np.ones(len(density), dtype=float))
            y = np.outer(np.ones(len(density[0]), dtype=float),
                     np.arange(-size+density_grid_len/2, R_max, density_grid_len, dtype=float))
            distance = (x**2+y**2)**(1/2)
            ## Flatten and sort by the distance from the origin
            ind = np.argsort(distance.flatten())
            density = np.take_along_axis(density.flatten(), ind, axis=0)
            x = np.take_along_axis(x.flatten(), ind, axis=0)
            y = np.take_along_axis(y.flatten(), ind, axis=0)
            
            ## Construct luminosity
            luminosity = np.zeros_like(R_domain, dtype=float)
            for j in range(len(luminosity)-1):
                for k in range(len(density)):
                    if distance[k]+density_grid_len*2**(1/2) < R_domain[j+1]:
                        luminosity[i+1] += density[k]
                    ## When only fraction of grid is included in circle of R
                    ## Approximate non-overlap region as a triangle
                    elif distance[k] < R_domain[j+1]:
                        x_intersect = (R_domain[j+1]**2-(abs(y[k])-density_grid_len/2)**2)**(1/2)
                        y_intersect = (R_domain[j+1]**2-(abs(x[k])-density_grid_len/2)**2)**(1/2)
                        overlap_x_len = density_grid_len/2-abs(x[k])+x_intersect
                        overlap_y_len = density_grid_len/2-abs(y[k])+y_intersect
                        not_overlap_frac = overlap_x_len*overlap_y_len/2/density_grid_len**2
                        luminosity[i+1] += density[k]*(1-not_overlap_frac)
                    ## Simliar to above, but this time approximate overlap region
                    elif distance[k]-density_grid_len*2**(1/2) < R_domain[j+1]:
                        x_intersect = (R_domain[j+1]**2-(abs(y[k])-density_grid_len/2)**2)**(1/2)
                        y_intersect = (R_domain[j+1]**2-(abs(x[k])-density_grid_len/2)**2)**(1/2)
                        overlap_x_len = density_grid_len/2-abs(x[k])+x_intersect
                        overlap_y_len = density_grid_len/2-abs(y[k])+y_intersect
                        overlap_frac = overlap_x_len*overlap_y_len/2/density_grid_len**2
                        luminosity[i+1] += density[k]*overlap_frac
                    ## Since distance is sorted, once this distance is reached
                    ## there are no remaining grid that should be included
                    else:
                        break

            ## Compute I_e, luminosity at effective radius
            for j in range(len(R_domain)):
                if not R_e > R_domain[j]:
                    R_e_index = j
                    break
            I_e = luminosity[R_e_index]
        
            ## Find best 'n' value by using least square method
            initial_guess = [I_e, R_e, 4]
            popt, pcov = curve_fit(sersic_profile, R_domain[1:], np.log10(luminosity[1:]),
                                   p0=initial_guess, bounds=([I_e*0.98,R_e*0.98,0.1],[I_e*1.02,R_e*1.02,20]))
            ## Exclude the first element, since it incorporate large error
            print("%s - Effective radius: %.3f kpc, Sersic index: %.2f" % (ptype_name[i], popt[1], popt[2]))
        
            ## To find best fit in non-log scale, uncomment below line
            #popt, pcov = curve_fit(sersic_profile, r_domain[1:], luminosity[1:], p0=initial_guess, bounds=([1,R_e*0.99,0.1],[np.inf,R_e*1.01,20]))
            plt.plot(R_domain[1:], luminosity[1:], 'b-', label='Luminosity')
            plt.plot(R_domain[1:], 10**sersic_profile(R_domain[1:], *popt), 'r-', label='Sersic fit')
            plt.yscale('log')
            plt.xlabel('R (kpc)')
            plt.ylabel('Luminosity')
            plt.title("%s Sersic profile fitting" % ptype_name[i])
            plt.legend()
            plt.show()
        
        ## Sersic fitting for disk components
        ## Load disk particle data and merge them, if exists
        disk_ptypes = [2,4]
        data = np.array([-1,-1,-1])  ## Just initialization, will be removed
        for ptype in disk_ptypes:
            if ptype in ptype_exist:
                ptype_index = np.where(np.array(ptype_exist)==ptype)[0][0]
                data_one_ptype = data_list[ptype_index]
                data = np.vstack((data, data_one_ptype))  
            else:
                disk_ptypes.remove(ptype)
        data = np.delete(data, 0, axis=0)
        if len(disk_ptypes) < 1:  ## Terminate when no disk particle exists
            return None

        ## Construct R_grid 
        x_axis, y_axis = plane
        R = (data[:,x_axis]**2 + data[:,y_axis]**2)**(1/2)
        R = np.sort(R)
        R_len = len(R)
        particle_cutoff = int(R_len*0.95)
        R_max = R[particle_cutoff] 
        R_domain = np.linspace(0, R_max, 101, endpoint=True)

        ## Effective radius can be computed directly by sorting R
        R_e = (R[R_len//2-1]+R[R_len//2])/2 if R_len%2 else R[R_len//2]

        density_grid_len = 0.02
        ## To deal with boundary (R==size) grid, include more than specified size
        density_grid_num = 2*round(size/density_grid_len)+2
        density = kernel_density_estimation_2D(sdir, snum, disk_ptypes, R_max,
                                               (0,1), density_grid_num, resolution=5, opt=False)
        x = np.outer(np.arange(-size+density_grid_len/2, R_max, density_grid_len, dtype=float),
                 np.ones(len(density), dtype=float))
        y = np.outer(np.ones(len(density[0]), dtype=float),
                 np.arange(-size+density_grid_len/2, R_max, density_grid_len, dtype=float))
        distance = (x**2+y**2)**(1/2)
        ## Flatten and sort by the distance from the origin
        ind = np.argsort(distance.flatten())
        density = np.take_along_axis(density.flatten(), ind, axis=0)
        x = np.take_along_axis(x.flatten(), ind, axis=0)
        y = np.take_along_axis(y.flatten(), ind, axis=0)
        
        ## Construct luminosity
        luminosity = np.zeros_like(R_domain, dtype=float)
        for j in range(len(luminosity)-1):
            for k in range(len(density)):
                if distance[k]+density_grid_len*2**(1/2) < R_domain[j+1]:
                    luminosity[i+1] += density[k]
                ## When only fraction of grid is included in circle of R
                ## Approximate non-overlap region as a triangle
                elif distance[k] < R_domain[j+1]:
                    x_intersect = (R_domain[j+1]**2-(abs(y[k])-density_grid_len/2)**2)**(1/2)
                    y_intersect = (R_domain[j+1]**2-(abs(x[k])-density_grid_len/2)**2)**(1/2)
                    overlap_x_len = density_grid_len/2-abs(x[k])+x_intersect
                    overlap_y_len = density_grid_len/2-abs(y[k])+y_intersect
                    not_overlap_frac = overlap_x_len*overlap_y_len/2/density_grid_len**2
                    luminosity[i+1] += density[k]*(1-not_overlap_frac)
                ## Simliar to above, but this time approximate overlap region
                elif distance[k]-density_grid_len*2**(1/2) < R_domain[j+1]:
                    x_intersect = (R_domain[j+1]**2-(abs(y[k])-density_grid_len/2)**2)**(1/2)
                    y_intersect = (R_domain[j+1]**2-(abs(x[k])-density_grid_len/2)**2)**(1/2)
                    overlap_x_len = density_grid_len/2-abs(x[k])+x_intersect
                    overlap_y_len = density_grid_len/2-abs(y[k])+y_intersect
                    overlap_frac = overlap_x_len*overlap_y_len/2/density_grid_len**2
                    luminosity[i+1] += density[k]*overlap_frac
                ## Since distance is sorted, once this distance is reached
                ## there are no remaining grid that should be included
                else:
                    break

        ## Compute I_e, luminosity at effective radius
        for j in range(len(R_domain)):
            if not R_e > R_domain[j]:
                R_e_index = j
                break
        I_e = luminosity[R_e_index]
    
        ## Find best 'n' value by using least square method
        initial_guess = [I_e, R_e, 4]
        popt, pcov = curve_fit(sersic_profile, R_domain[1:], np.log10(luminosity[1:]),
                               p0=initial_guess, bounds=([I_e*0.98,R_e*0.98,0.1],[I_e*1.02,R_e*1.02,20]))
        ## Exclude the first element, since it incorporate large error
        print("%s - Effective radius: %.3f kpc, Sersic index: %.2f" % (ptype_name[i], popt[1], popt[2]))
    
        ## To find best fit in non-log scale, uncomment below line
        #popt, pcov = curve_fit(sersic_profile, r_domain[1:], luminosity[1:], p0=initial_guess, bounds=([1,R_e*0.99,0.1],[np.inf,R_e*1.01,20]))
        plt.plot(R_domain[1:], luminosity[1:], 'b-', label='Luminosity')
        plt.plot(R_domain[1:], 10**sersic_profile(R_domain[1:], *popt), 'r-', label='Sersic fit')
        plt.yscale('log')
        plt.xlabel('R (kpc)')
        plt.ylabel('Luminosity')
        plt.title("%s Sersic profile fitting" % ptype_name[i])
        plt.legend()
        plt.show()
        
    else:
        ## Load the locations and masses of the particles
        ptype_exist, data_list = load_data('Coordinates', [1,2,3,4], sdir, snum)
        masses = np.zeros_like(ptype_exist, dtype=float)
        for i in range(len(ptype_exist)):
            ptype = ptype_exist[i]
            mass = load_from_snapshot('Masses', ptype, sdir, snum)
            masses[i] = mass[0]  ## Assume mass of each ptype particle is the same
    
        ## All particles in halo or bulge components are uniform
        ## So Sersic fitting of those components are fairly easy
        ptype_name = ["Halo", "Bulge"]
        for i in range(len(ptype_name)):
    
            ptype = 2*i+1
            ## Continue to next ptype if ptype does not exist
            if ptype not in ptype_exist:
                break
            ptype_index = np.where(np.array(ptype_exist) == ptype)[0][0]
            data = data_list[ptype_index]
                
            ## Compute R_e, where half of the particles reside inside
            x_axis, y_axis = plane
            R = (data[:,x_axis]**2 + data[:,y_axis]**2)**(1/2)
            R = np.sort(R)
            R_len = len(R)
            ## Effective radius can be computed directly by sorting R
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
                density[j] = particle_per_grid/area*masses[ptype_index]
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
            initial_guess = [I_e, R_e, 4]
            popt, pcov = curve_fit(sersic_profile, r_domain[1:], np.log10(luminosity[1:]),
                                   p0=initial_guess, bounds=([I_e*0.98,R_e*0.98,0.1],[I_e*1.02,R_e*1.02,20]))
            ## Exclude the first element, since it incorporate large error
            print("%s - Effective radius: %.3f kpc, Sersic index: %.2f" % (ptype_name[i], popt[1], popt[2]))
        
            ## To find best fit in non-log scale, uncomment below line
            #popt, pcov = curve_fit(sersic_profile, r_domain[1:], luminosity[1:], p0=initial_guess, bounds=([1,R_e*0.99,0.1],[np.inf,R_e*1.01,20]))
            plt.plot(r_domain[1:], luminosity[1:], 'b-', label='Luminosity')
            plt.plot(r_domain[1:], 10**sersic_profile(r_domain[1:], *popt), 'r-', label='Sersic fit')
            plt.yscale('log')
            plt.xlabel('R (kpc)')
            plt.ylabel('Luminosity')
            plt.title("%s Sersic profile fitting" % ptype_name[i])
            plt.legend()
            plt.show()
        
        ## For disk Sersic fitting, newly formed stars should be included
        ## Fist examine existence of disk
        disk_ptypes = [2,4]
        for ptype in disk_ptypes:
            if ptype not in ptype_exist:
                disk_ptypes.remove(ptype)
        
        if len(disk_ptypes) > 0:
            ## Constuct data list including every disk ptypes and their masses
            data = np.array([-1,-1,-1,-1])  ## Just initialization, will be removed
            for ptype in disk_ptypes:
                ptype_index = np.where(np.array(ptype_exist)==ptype)[0][0]
                data_one_ptype = data_list[ptype_index]
                mass_one_ptype = np.ones_like(data_one_ptype[:,0])*masses[ptype_index]
                data_one_ptype = np.hstack((data_one_ptype, mass_one_ptype[:,None]))
                data = np.vstack((data, data_one_ptype))
            data = np.delete(data, 0, axis=0)
        
            ## Compute R_e, where half of the particles reside inside
            x_axis, y_axis = plane
            R = (data[:,x_axis]**2 + data[:,y_axis]**2)**(1/2)
            mass = data[:,3]
            ind = np.argsort(R, axis=0)
            R = np.take_along_axis(R, ind, axis=0)
            mass = np.take_along_axis(mass, ind, axis=0)
            half_mass = np.sum(mass)/2
            mass_within_R = 0
            for i in range(len(R)):
                mass_within_R += mass[i]
                if mass_within_R > half_mass:
                    break
            ## Interplate to compute R_e
            R_e = R[i] - (R[i]-R[i-1])*(mass_within_R-half_mass)/mass[i]
    
            ## Divide r domain so that each r grid contains certain number of particles
            particle_cutoff = int(len(R)*0.95)  ## Exclude outliers
            resolution = 100
            particle_per_grid = int(particle_cutoff//resolution)
            r_domain_index = np.arange(0, particle_cutoff+1, particle_per_grid)
            r_domain = R[r_domain_index]
        
            ## Compute luminosity (=particle number density) as a function of R
            density = np.zeros(len(r_domain)-1)
            for i in range(len(density)):
                area = np.pi*(r_domain[i+1]**2 - r_domain[i]**2)
                for j in range(particle_per_grid):
                    density[i] += mass[i*particle_per_grid+j]
                density[i] /= area
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
                                   p0=initial_guess, bounds=([I_e*0.98,R_e*0.98,0.1],[I_e*1.02,R_e*1.02,20]))
            ## Exclude the first element, since it incorporate large error
            print("Disk - Effective radius: %.3f kpc, Sersic index: %.2f" % (popt[1], popt[2]))
        
            ## To find best fit in non-log scale, uncomment below line
            #popt, pcov = curve_fit(sersic_profile, r_domain[1:], luminosity[1:], p0=initial_guess, bounds=([1,R_e*0.99,0.1],[np.inf,R_e*1.01,20]))
            plt.plot(r_domain[1:], luminosity[1:], 'b-', label='Luminosity')
            plt.plot(r_domain[1:], 10**sersic_profile(r_domain[1:], *popt), 'r-', label='Sersic fit')
            plt.yscale('log')
            plt.xlabel('R (kpc)')
            plt.ylabel('Luminosity')
            plt.title("Disk Sersic profile fitting")
            plt.legend()
            plt.show()


    return None


def azimuthal_structure(sdir, snum, size=6., kernel=True):
    """This subrutine computes radial density function of the disk, then take 
    Fourier transform to the density function and plot the results.
    
    Arguments:
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file
            
      snum (int): number of the snapshot. e.g. snapshot_001.hdf5 is '1'
        Note for multi-part files, this is just the number of the 'set', i.e. 
        if you have snapshot_001.N.hdf5, set this to '1', not 'N' or '1.N'

      size (float): The size of the region that you will analyze in code unit
        (nomarlly kpc - but inspect your GIZMO setting!).

    Optionals:
      kernel (boolean): Choose whether use kernel density estimation or not.
        Using kernel density estimation is more accurate, but slower.
        
    Raises:
      If there are no disk particles, the entire program will be halted.

    Returns: Nothing
    """

    ## This one is slower since it uses kernel estimation, but more accurate
    if kernel:

        ## Set grid points for FFT analysis
        R_grid_len = 0.01
        R_grid = np.arange(0, size+R_grid_len/2, R_grid_len)
        theta_grid_num = 50
        theta_grid = np.linspace(0, 2*np.pi, theta_grid_num, endpoint=False)

        ## Create density matrix using kernel density estimation
        disk_ptype_exist, data_list = load_data('Coordinates', [2,4], sdir, snum,
                                          find_center=True)
        density_grid_len = 0.02
        ## To deal with boundary (R==size) grid, include more than specified size
        density_grid_num = 2*round(size/density_grid_len)+2
        density = kernel_density_estimation_2D(sdir, snum, disk_ptype_exist, size+density_grid_len,
                                               (0,1), density_grid_num, resolution=5, opt=False)
        ## Resolution is chosen to make the result similar to the reference,
        ## arXiv:1901.02021

        ## Construct FFT results matrix
        mode_min, mode_max= 1, 5  ## Maximum and minimum mode of FFT
        FFT_results = np.zeros((len(R_grid), mode_max-mode_min+1))
        for i in range(len(R_grid)):
            if i == 0:
                continue
            R = R_grid[i]
            density_each_R = np.zeros_like(theta_grid)
            for j in range(len(theta_grid)):
                theta = theta_grid[j]
                ## Assign index to find neighbors
                x, y = R*np.cos(theta), R*np.sin(theta)
                x_index = (x+size)/density_grid_len
                y_index = (y+size)/density_grid_len
                ## Bilinear interpolation from neighboring grids
                x_down, y_down = int(x_index), int(y_index)
                x_up, y_up = x_down+1, y_down+1
                x_temp = np.array([x_up-x_index, x_index-x_down])
                y_temp = np.array([y_up-y_index, y_index-y_down])
                neighbors_temp = np.array([[density[x_down, y_down], density[x_down, y_up]],
                                           [density[x_up  , y_down], density[x_up  , y_up]]])
                density_each_R[j] = np.dot(np.dot(x_temp, neighbors_temp), y_temp)/R_grid_len**2
            FFT_each_R = np.fft.fft(density_each_R)
            FFT_each_R /= FFT_each_R[0]
            FFT_results[i] = abs(FFT_each_R[mode_min: mode_max+1])

        ## Plot the results
        mode_domain = np.arange(mode_min, mode_max+1, 1, dtype=int)
        for mode in mode_domain:
            plt.plot(R_grid, abs(FFT_results[:,mode-mode_min]), label="m=%i" %mode)
        time = snum*0.01
        plt.title("t=%.2f Gyr azimuthal mode analysis" %time)
        plt.xlabel("R (kpc)")
        plt.ylabel("a_m/a_0")
        plt.legend()
        plt.show()

    ## This one is much faster, but inaccurate since it does not use kernel
###########################################################################
# It seems like useless (too inacurrate). It is highly probable to delete #
# this code in the near future.                                           #
###########################################################################    
    else:
        ## Load the locations of the particles
        ptype_list, data_list = load_data('Coordinates', [2,4], sdir, snum,
                                          find_center=True)
    
        ## Compute R for each particle
        data = data_list[0]
        if len(data_list) > 1:  ## For no-star formation case
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
        theta_grid_num = 50  ## Beyond 30, the result is almost stable
        theta_grid = np.linspace(0, 2*np.pi, theta_grid_num, endpoint=False)
        theta_grid_len = 2*np.pi/theta_grid_num
        theta_grid_point = theta_grid + theta_grid_len/2
    
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
                while (theta_index < theta_index_max
                       and R_grid_data[theta_index, 1] < theta_grid[i+1]):
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


def measure_SFR(sdir, snum_max, save=False):
    """Measure the star formation rate (SFR) of the entire galaxy by plotting
    the number of gas and newly created star particles.

    Arguments:
      sdir (string): parent directory of the snapshot file or immediate
        snapshot sub-directory if it is a multi-part file
            
      snum_max (int): Maximum number of the snapshot.

    Optionals:
      save (boolean): Determine whether save the plot or not. Default is False.
        You can specify the saving directory. Default will be a sub-directory
        of the data directory, and if the directory does not exist,
        then it will create the directory.grid_num=4

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
    plt.ylim(0, 10000)
    plt.xlim(0,1)
    lns1 = ax1.plot(t_list, num_ptype0_list, 'c-', label='ptype0')
    lns2 = ax1.plot(t_list, num_ptype4_list, 'm-', label='ptype4')
    ax2 = ax1.twinx()
    ax2.set_ylabel('SFR (M_sun/yr)')
    lns3 = ax2.plot(t_list, SFR_list, 'r-', label='SFR')
    plt.ylim(0, 50)
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    fig.tight_layout()
    plt.title('Time vs ptype0, ptype4 number and SFR')
    ## Save the figure if save flag is on
    if save:
        ## Create the directory if desniated directory does not exist
        Path(sdir+"/plot").mkdir(parents=True, exist_ok=True)
        plt.savefig(sdir+"/plot/SFR.png")
        
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
    ptype_list, data_list = load_data(value, ptype_list, sdir, snum)

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


def measure_minimum_distance(sdir, snum, ptype_list, size, num_dist):
    
    dummy, data_list = load_data('Coordinates', ptype_list, sdir, snum)
    dummy, id_list = load_data('ParticleIDs', ptype_list, sdir, snum)
    ## Extract particles only within cube of size
    data_merged = np.zeros(3)
    id_merged = np.zeros(1)
    for i in range(len(data_list)):
        data, ids = data_list[i], id_list[i]
        ids = ids[np.where((abs(data[:,0]) < size)
                            & (abs(data[:,1]) < size)
                            & (abs(data[:,2]) < size))]
        id_merged = np.append(id_merged, ids)
        data = data[np.where((abs(data[:,0]) < size)
                             & (abs(data[:,1]) < size)
                             & (abs(data[:,2]) < size))]
        data_merged = np.vstack((data_merged, data))
    data_merged = np.delete(data_merged, 0, axis=0)
    id_merged = np.delete(id_merged, 0, axis=0)
    
    min_dist_list = np.array([10000000])  ## Arbitrary large dummy number
    min_dist_ids = np.zeros(2)
    print(len(data_merged))
    for i in range(len(data_merged)-1):

        loc1, id1 = data_merged[i], id_merged[i]
        j = i+1
        while j < len(data_merged):
            loc2, id2 = data_merged[j], id_merged[j]
            dist = np.sum((loc1-loc2)**2)**(1/2)
            if dist < min_dist_list[-1]:  ## Last element is the biggest since sorted
                min_dist_list = np.append(min_dist_list, dist)
                min_dist_ids = np.vstack((min_dist_ids, np.array([id1, id2])))
                ind = np.argsort(min_dist_list)
                min_dist_list = np.take_along_axis(min_dist_list, ind, axis=0)
                min_dist_ids[:,0] = np.take_along_axis(min_dist_ids[:,0], ind, axis=0)
                min_dist_ids[:,1] = np.take_along_axis(min_dist_ids[:,1], ind, axis=0)
                
                if len(min_dist_list) > num_dist:
                    min_dist_list = np.delete(min_dist_list, -1, axis=0)
                    min_dist_ids = np.delete(min_dist_ids, -1, axis=0)
            j += 1
    
    return min_dist_list, min_dist_ids
            

## Select model
#sdir = "/home/du/GIZMO/AddGas/temp"
#sdir = "/home/du/GIZMO/GIZMO_Test/Before_Update/Model_Bar1/CoolingLowT/results"
#sdir = "/home/du/GIZMO/GIZMO_Test/Before_Update/Model_Bar1_Feedback/Scale_2_Orientation_0/results"
sdir = "/home/du/GIZMO/Meeting/OnlyGrav/results"
snum = 100
ptype_list = [0]
size = 20
plane = (0,2)

## Plot the locations of the particles
plot_particles(sdir, snum, ptype_list, size, plane, find_center=False, trackID=0, save=True)

## Plot the density contours
#a = time.time()
#plot_density_contour(sdir, snum, ptype_list, size, plane,
#                     grid_num=400, save=False, opt=False)
#print(time.time()-a)

## Fit the luminosity curve to Sersic profile
#sersic_fitting(sdir, snum, plane)

## Conduct Fourier analysis of the density of the disk to investigate bar and spiral structures
#snum_list = np.arange(0, 101,1)
#for snum in snum_list:
#    plot_particles(sdir, snum, ptype_list, size, plane, trackID=0, find_center=False, save=False)
#    azimuthal_structure(sdir, snum, 6., kernel=True)

## Investigate SFR
#max_snum = 100
#measure_SFR(sdir, max_snum, save=True)

#track_IDs = np.arange(201000, 201100,1)
#for temp in track_IDs:
#    plot_particles(sdir, snum, sizes=2, axes=[(0,2)], ptype_list=[0], trackID=temp, save=False)

## Plot the evolution consequnce of the galaxy
#snum_list = np.arange(0,101,1)
#for snum in snum_list:
#    plot_particles(sdir, snum, ptype_list, size, plane, trackID=0, find_center=False, save=True)
#    plot_density_contour(sdir, snum, ptype_list, size, plane,
#                         grid_num=400, save=False, opt=False)

## Find the minimum distances between particles
#temp, ids = measure_minimum_distance(sdir, snum, ptype_list, size, 15)

## Compare rotatoin curve of disk particles and gas particls
#dummy, loc = load_data('Coordinates', [0,2], sdir, snum)
#dummy, vel = load_data('Velocities', [0,2], sdir, snum)
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

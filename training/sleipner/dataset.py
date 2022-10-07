#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from distdl.backends.common.partition import MPIPartition
import distdl.utilities.slicing as slicing
from torch.utils.data import Dataset 
from distdl.utilities.torch import *
from mpi4py import MPI
import azure.storage.blob
import h5py, zarr, os
import numpy as np 
import torch


class SleipnerSerial(Dataset):
    ''' Dataset class for flow data generated with OPM 
    This dataset class repeats 3D models in the temporal dimension
    '''

    def __init__(self, samples, client, container, prefix, shape, normalize=True, padding=None,
        savepath=None, filename=None, keep_data=False, target='saturation'):
        """ Pytorch dataset class for Sleipner data set.
        """

        self.samples = samples
        self.client = client
        self.container = container
        self.prefix = prefix
        self.normalize = normalize
        self.padding = padding
        self.shape = shape
        self.savepath = savepath
        self.keep_data = keep_data
        self.filename = filename
        if target == 'saturation':
            target = 1
        elif target == 'pressure':
            target = 0
        self.target = target
        if savepath is not None:
            self.cache = list()
            # Check if files were already downloaded
            files = os.listdir(savepath)
            for i in samples:
                if filename + '_' + str(int(i.item())) + '.h5' in files:
                    self.cache.append(self.filename + '_' + str(i.item()) + '.h5')
        else:
            self.cache = None

        # Open the data file
        self.store = zarr.ABSStore(container=self.container, prefix=self.prefix, client=self.client)       

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        # Read 
        i = int(self.samples[index])

        # If caching is used, check if data sample exists locally
        if self.cache is not None and self.filename + '_' + str(i) + '.h5' in self.cache:
            fid = h5py.File(os.path.join(self.savepath, self.filename + '_' + str(i) + '.h5'), 'r')
            x = torch.tensor(np.array(fid['x']))
            y = torch.tensor(np.array(fid['y']))
            nt = y.shape[-1]
            fid.close()

        else:

            # Static parameters
            permx = torch.tensor(np.array(zarr.core.Array(self.store, path='permx')), dtype=torch.float32)
            permz = torch.tensor(np.array(zarr.core.Array(self.store, path='permz')), dtype=torch.float32)
            depth = torch.tensor(np.array(zarr.core.Array(self.store, path='depth')), dtype=torch.float32)

            # Dynamic parameters (Z Y X T)
            wellmap = torch.tensor(np.array(zarr.core.Array(self.store, path='well_' + str(i))), dtype=torch.float32)
            pressure = torch.tensor(np.array(zarr.core.Array(self.store, path='pressure_' + str(i))), dtype=torch.float32)
            saturation = torch.tensor(np.array(zarr.core.Array(self.store, path='saturation_' + str(i))), dtype=torch.float32)
            nx, ny, nz, nt = saturation.shape

            # Normalize
            if self.normalize:
                permx -= permx.min(); permx /= permx.max()
                permz -= permz.min(); permz /= permz.max()
                depth -= depth.min(); depth /= depth.max()
                pressure -= 130; pressure /= 400
                saturation[saturation < 0] = 0; #saturation /= saturation.max()

            # Reshape to [ C X Y Z T]
            permx = permx.view(1, nx, ny, nz, 1,)
            permz = permz.view(1, nx, ny, nz, 1)
            depth = depth.view(1, nx, ny, nz, 1)
            wellmap = wellmap.view(1, 1, ny, nz, 1).repeat(1, nx, 1, 1, 1)
            pressure = pressure.view(1, nx, ny, nz, nt)        # C=1 X Y Z T 
            saturation = saturation.view(1, nx, ny, nz, nt)    # C=1 X Y Z T

            x = torch.cat((
                permx,
                permz,
                depth,
                wellmap
                ),
                axis=0
            )
            
            y = torch.cat((
                pressure,
                saturation
                ),
                axis=0
            )

            if self.cache is not None:
                fid = h5py.File(os.path.join(self.savepath, self.filename + '_' + str(i) + '.h5'), 'w')
                fid.create_dataset('x', data=x)
                fid.create_dataset('y', data=y)
                fid.close()
                self.cache.append(self.filename + '_' + str(i) + '.h5')

        return x.repeat(1, 1, 1, 1, nt), y[self.target:self.target+1,:,:,:,:]
        

    def close(self):
        if self.keep_data is False and self.cache is not None:
            print('Delete temp files.')
            for file in self.cache:
                os.system('rm ' + self.savepath + '/' + file)


###################################################################################################
# Parallel dataloader

class SleipnerParallel(Dataset):
    ''' Distributed Dataset class for flow data generated with OPM'''

    def __init__(self, P_feat, samples, client, container, prefix, shape, normalize=True, 
        savepath=None, filename=None, keep_data=False, target='saturation'):
        
        self.P_feat = P_feat
        self.samples = samples
        self.client = client
        self.container = container
        self.prefix = prefix
        self.normalize = normalize
        self.savepath = savepath
        self.shape = shape
        self.keep_data = keep_data
        self.filename = filename
        self.yStart = slicing.compute_start_index(P_feat.shape[2:], P_feat.index[2:], self.shape)[1]
        self.yEnd = slicing.compute_stop_index(P_feat.shape[2:], P_feat.index[2:], self.shape)[1]
        if target == 'saturation':
            target = 1
        elif target == 'pressure':
            target = 0
        self.target = target
        if savepath is not None:
            self.cache = list()

            # Check if files were already downloaded
            files = os.listdir(savepath)
            for i in samples:
                filename_curr = f'{self.filename}_{i:04d}_{self.P_feat.rank:04d}.h5'
                if filename_curr in files:
                    self.cache.append(filename_curr)
        else:
            self.cache = None
        
        # Open the data file
        self.store = zarr.ABSStore(container=self.container, prefix=self.prefix, client=self.client)       

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        # Read 
        i = int(self.samples[index])
        
        # If caching is used, check if data sample exists locally
        filename = f'{self.filename}_{i:04d}_{self.P_feat.rank:04d}.h5'
        if self.cache is not None and filename in self.cache:
            fid = h5py.File(os.path.join(self.savepath, filename), 'r')
            x = torch.tensor(np.array(fid['x']))
            y = torch.tensor(np.array(fid['y']))
            nt = y.shape[-1]
            fid.close()

        else:

            # Static parameters
            permx = torch.tensor(np.array(zarr.core.Array(self.store, path='permx')[1:,self.yStart:self.yEnd,:]), dtype=torch.float32)
            permz = torch.tensor(np.array(zarr.core.Array(self.store, path='permz')[1:,self.yStart:self.yEnd,:]), dtype=torch.float32)
            depth = torch.tensor(np.array(zarr.core.Array(self.store, path='depth')[1:,self.yStart:self.yEnd,:]), dtype=torch.float32)

            wellmap = torch.tensor(np.array(zarr.core.Array(self.store, path='well_' + str(i))[self.yStart:self.yEnd,:]), dtype=torch.float32)
            pressure = torch.tensor(np.array(zarr.core.Array(self.store, path='pressure_' + str(i))[1:,self.yStart:self.yEnd,:,:]), dtype=torch.float32)
            saturation = torch.tensor(np.array(zarr.core.Array(self.store, path='saturation_' + str(i))[1:,self.yStart:self.yEnd,:,:]), dtype=torch.float32)
                    
            # Normalize between 0 and 1
            saturation[saturation < 0] = 0
            if self.normalize:
                pressure -= 130; pressure /= 400
                permx -= 1.2000e-05; permx /= 43.5386
                permz -= 1.2000e-5; permz /= 2495.3101
                depth -= 728.0043; depth /= 363.9501
            
            # Reshape to [ C X Y Z T]
            nx, ny, nz, nt = saturation.shape
            permx = permx.view(1, nx, ny, nz, 1)
            permz = permz.view(1, nx, ny, nz, 1)
            depth = depth.view(1, nx, ny, nz, 1)
            wellmap = wellmap.view(1, 1, ny, nz, 1).repeat(1, nx, 1, 1, 1)
            pressure = pressure.view(1, nx, ny, nz, nt)        # C=1 X Y Z T
            saturation = saturation.view(1, nx, ny, nz, nt)    # C=1 X Y Z T

            x = torch.cat((
                permx,
                permz,
                depth,
                wellmap
                ),
                axis=0
            )
            
            y = torch.cat((
                pressure,
                saturation
                ),
                axis=0
            )

            # Write file to disk for later use
            if self.cache is not None:
                fid = h5py.File(os.path.join(self.savepath, filename), 'w')
                fid.create_dataset('x', data=x)
                fid.create_dataset('y', data=y)
                fid.close()
                self.cache.append(filename)

        return x.repeat(1, 1, 1, 1, nt), y[self.target:self.target+1,:,:,:,:]


    def close(self):
        if self.keep_data is False and self.cache is not None:
            print('Delete temp files.')
            for file in self.cache:
                os.system('rm ' + self.savepath + '/' + file)
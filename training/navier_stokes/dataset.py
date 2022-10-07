#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import distdl.utilities.slicing as slicing
from distdl.utilities.torch import *
from distdl.backends.common.partition import MPIPartition
from torch.utils.data import Dataset 
from mpi4py import MPI
import azure.storage.blob
import h5py, zarr, os
import numpy as np 
import torch


###################################################################################################
# Serial dataloader


class WaterlilySerial(Dataset):
    ''' Distributed Dataset class for Navier Stokes data generated with WaterLily.jl'''


    def __init__(self, samples, client, container, prefix, shape, normalize=True, 
        savepath=None, filename=None, keep_data=None, clip=None, target='vorticity'):

        self.samples = samples
        self.client = client
        self.container = container
        self.prefix = prefix
        self.shape = shape
        self.normalize = normalize
        self.savepath = savepath
        self.keep_data = keep_data
        self.filename = filename
        self.clip = clip
        
        if savepath is not None:
            self.cache = list()

            # Check if files were already downloaded
            files = os.listdir(savepath)
            for i in samples:
                filename_curr = f'{self.filename}_{i:04d}.h5'
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
        filename = f'{self.filename}_{i:04d}.h5'
        if self.cache is not None and filename in self.cache:
            fid = h5py.File(os.path.join(self.savepath, filename), 'r')
            x = torch.tensor(np.array(fid['x']))
            y = torch.tensor(np.array(fid['y']))
            nt = y.shape[-1]
            fid.close()

        else:

            x = torch.tensor(np.array(zarr.core.Array(self.store, path='model_' + str(i))[:,:,:,:3]), dtype=torch.float32)
            vorticity = torch.tensor(np.array(zarr.core.Array(self.store, path='vorticity_' + str(i))[:,:,:,1:]), dtype=torch.float32)
            nc_in = x.shape[-1]

            # Normalize
            if self.normalize:
                x -= x.min(); x /= x.max()
                vorticity -= vorticity.min(); vorticity /= vorticity.max() / 2
                vorticity -= 1

            # Clip amplitudes
            if self.clip is not None:
                vorticity[vorticity > self.clip] = self.clip
                vorticity[vorticity < -self.clip] = -self.clip

                if self.normalize:  # renormalize
                    vorticity = vorticity / torch.max(torch.abs(vorticity))

            # Reshape to [ C X Y Z T]
            nx, ny, nz, nt = self.shape
            x = x.permute(3, 0, 1, 2).view(nc_in, nx, ny, nz, 1)   
            vorticity = vorticity.view(1, nx, ny, nz, nt)
            y = vorticity

            # Write file to disk for later use
            if self.cache is not None:
                fid = h5py.File(os.path.join(self.savepath, filename), 'w')
                fid.create_dataset('x', data=x)
                fid.create_dataset('y', data=y)
                fid.close()
                self.cache.append(filename)

        # Repeat x along time axis
        return x.repeat(1, 1, 1, 1, nt), y
        


###################################################################################################
# Parallel dataloader

class WaterlilyParallel(Dataset):
    ''' Distributed Dataset class for Navier Stokes data generated with WaterLily.jl'''


    def __init__(self, P_feat, samples, client, container, prefix, shape, normalize=True, 
        savepath=None, filename=None, keep_data=None, clip=None, target='vorticity'):

        self.P_feat = P_feat
        self.samples = samples
        self.client = client
        self.container = container
        self.prefix = prefix
        self.shape = shape
        self.normalize = normalize
        self.savepath = savepath
        self.keep_data = keep_data
        self.filename = filename
        self.clip = clip
        self.yStart = slicing.compute_start_index(P_feat.shape[2:], P_feat.index[2:], self.shape)[1]
        self.yEnd = slicing.compute_stop_index(P_feat.shape[2:], P_feat.index[2:], self.shape)[1]
        
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

            x = torch.tensor(np.array(zarr.core.Array(self.store, path='model_' + str(i))[:,self.yStart:self.yEnd,:,:3]), dtype=torch.float32)
            vorticity = torch.tensor(np.array(zarr.core.Array(self.store, path='vorticity_' + str(i))[:,self.yStart:self.yEnd,:,1:]), dtype=torch.float32)
            nx, ny, nz, nc_in = x.shape
            nt = vorticity.shape[-1]

            # Clip amplitudes
            if self.clip is not None:
                vorticity[vorticity > self.clip] = self.clip
                vorticity[vorticity < -self.clip] = -self.clip

                if self.normalize:  # renormalize
                    vorticity = vorticity / self.clip

            # Reshape to [ C X Y Z T]
            x = x.permute(3, 0, 1, 2).view(nc_in, nx, ny, nz, 1)   
            vorticity = vorticity.view(1, nx, ny, nz, nt)
            y = vorticity

            # Write file to disk for later use
            if self.cache is not None:
                fid = h5py.File(os.path.join(self.savepath, filename), 'w')
                fid.create_dataset('x', data=x)
                fid.create_dataset('y', data=y)
                fid.close()
                self.cache.append(filename)

        # Repeat x along time axis
        return x.repeat(1, 1, 1, 1, nt), y
        

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from distdl.backend import BackendProtocol, FrontEndProtocol, ModelProtocol, init_distdl 
from distdl.backends.common.partition import MPIPartition
from pfno import ParallelFNO4d, DistributedRelativeLpLoss
from distdl.utilities.torch import zero_volume_tensor
from distdl.nn.repartition import Repartition
import distdl, torch, os, time, h5py
from mpi4py import MPI
import numpy as np

# Set comm backend
init_distdl(frontend_protocol=FrontEndProtocol.MPI,
            backend_protocol=BackendProtocol.NCCL,
            model_protocol=ModelProtocol.CUPY)

# Init MPI
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()
n = P_world.shape[0]

# Master worker partition with 6 dimensions ( N C X Y Z T )
root_shape = (1, 1, 1, 1, 1, 1)
P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition(root_shape)

# Distributed paritions
feat_workers = np.arange(0, n)
P_feat_base = P_world.create_partition_inclusive(feat_workers)
P_data = P_feat_base.create_cartesian_topology_partition((1,1,1,n,1,1))    # Partition in physical space - data/weights are distributed along the y dimension
P_spec = P_feat_base.create_cartesian_topology_partition((1,1,n,1,1,1))    # Partition in Fourier space - data/weights are distributed along the x dimension

# Data dimensions
n_batch = 1
shape = (64, 64, 64, 64)    # Grid points in X Y Z T

# Network dimensions
channel_in = 4
channel_hidden = 8
channel_out = 2
num_k = (4, 4, 4, 4)    # Fourier modes in X Y Z T

# FNO
t0 = time.time()
pfno = ParallelFNO4d(
    P_world, 
    P_root,
    P_data,
    P_spec,
    channel_in,
    channel_hidden,
    channel_out,
    num_k
)

# Training
parameters = [p for p in pfno.parameters()]

# Optimizer
if len(parameters) > 0:
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
else:
    optimizer = None

# Move model to GPU
pfno = pfno.to(P_data.device)
pfno.train()

# Loss function
criterion = DistributedRelativeLpLoss(P_root, P_data).to(P_data.device)

# Generate random input/output data. Here we generate the data on the master worker (gpu0) 
# and distributed it to the remaining GPUs using the Repartition operator.
# When using a torch data loader, we can either read each full sample on GPU0 and distribute it,
# or we have each GPU read its y-slice of the data. See the data loaders in the sleipner and 
# navier_stokes directory for an example of this.
if P_root.active:
    x = torch.randn(n_batch, channel_in, *shape, device=P_data.device)
    y = torch.randn(n_batch, channel_out, *shape, device=P_data.device)
else:
    x = zero_volume_tensor(device=P_data.device)
    y = zero_volume_tensor(device=P_data.device)
print("Before scattering -> x.shape: {}; x.device: {}".format(y.shape, y.device))   # data shape is [0] on all workers except on GPU0

# Distribute data on P_data partition, i.e. split data along the y dimension
scatter_input = Repartition(P_root, P_data, preserve_batch=True)
scatter_output = Repartition(P_root, P_data, preserve_batch=True)
x = scatter_input(x)
y = scatter_output(y)
print("After scattering -> x.shape: {}; x.device: {}".format(y.shape, y.device))    # data is equally distributed among workers

if optimizer is not None:
    optimizer.zero_grad()

# Forward pass
y_ = pfno(x)

# Compute loss
loss = criterion(y_, y)

# Backward pass
loss.backward()
if optimizer is not None:
    optimizer.step()

# Collect output tensor on one GPU (e.g. to save it)
collect_output = Repartition(P_data, P_root, preserve_batch=False)
print("Before collection -> y_.shape: {}; y_.device: {}".format(y_.shape, y_.device))
y_ = collect_output(y_)
print("After collection -> y_.shape: {}; y_.device: {}".format(y_.shape, y_.device))
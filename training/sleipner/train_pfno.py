#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from distdl.backend import BackendProtocol, FrontEndProtocol, ModelProtocol, init_distdl
from pfno import ParallelFNO4d, DistributedRelativeLpLoss
from distdl.backends.common.partition import MPIPartition
import numpy as np
import cupy as cp
import distdl, torch, os, time, h5py
from mpi4py import MPI
import azure.storage.blob
from dataset import SleipnerParallel

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
P_x = P_feat_base.create_cartesian_topology_partition((1,1,1,n,1,1))
P_y = P_feat_base.create_cartesian_topology_partition((1,1,n,1,1,1))

# Reproducibility
torch.manual_seed(P_x.rank + 123)
np.random.seed(P_x.rank + 123)

# Data dimensions
nb = 2
shape = (262, 118, 64, 86)    # X Y Z T
num_train = 1400
num_valid = 176

# Network dimensions
channel_in = 4
channel_hidden = 10
channel_out = 1
num_k = (4, 4, 4, 4)

# Data store
container = os.environ['CONTAINER']
data_path = os.environ['DATA_PATH']

client = azure.storage.blob.ContainerClient(
    account_url=os.environ['ACCOUNT_URL'],
    container_name=container,
    credential=os.environ['CREDENTIALS']
    )

# Training dataset
train_idx = torch.linspace(1, num_train, num_train, dtype=torch.int32).long()   # missing: 404
train_data = SleipnerParallel(P_x, train_idx, client, container, data_path, shape, savepath=os.environ['DATA_DIR'], filename='sleipner')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=nb, shuffle=False)

# Validation dataset
valid_idx = torch.linspace(1 + num_train, num_train + num_valid, num_valid, dtype=torch.int32).long()
valid_data = SleipnerParallel(P_x, valid_idx, client, container, data_path, shape, savepath=os.environ['DATA_DIR'], filename='sleipner')
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=nb, shuffle=False)
P_world._comm.Barrier()

# FNO
t0 = time.time()
pfno = ParallelFNO4d(
    P_world, 
    P_root,
    P_x,
    P_y,
    channel_in,
    channel_hidden,
    channel_out,
    num_k
)

# Training
out_dir = os.environ['MODEL_DIR']
parameters = [p for p in pfno.parameters()]

# Optimizer
if len(parameters) > 0:
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
else:
    optimizer = None

# Restart from checkpoint?
start_epoch = 0
if start_epoch > 0:

    # Load model and optimizer state
    checkpoint_path = os.path.join(out_dir, f'model_snapshot_{start_epoch:04d}_{P_x.rank:04d}.pt')
    checkpoint = torch.load(checkpoint_path)
    pfno.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load loss
    if P_root.active:
        lossname = 'loss_epoch_' + str(epoch) + '.h5'
        fid = h5py.File(os.path.join(out_dir, lossname), 'r')
        train_accs = list(fid['train_loss'])
        valid_accs = list(fid['valid_loss'])
        fid.close()
else:
    epoch = -1
    if P_root.active:
        train_accs = []
        valid_accs = []

# Move model to GPU
pfno = pfno.to(P_x.device)

# Training loop
num_epochs = 50
checkpoint_interval = 1
criterion = DistributedRelativeLpLoss(P_root, P_x).to(P_x.device)

for i in range(epoch+1, num_epochs):

    # Loop over training data
    pfno.train()
    train_loss = 0
    n_train_batch = 0
    t0 = time.time()

    for j, (x, y) in enumerate(train_loader):
        
        x = x.to(P_x.device)
        y = y.to(P_x.device)

        if optimizer is not None:
            optimizer.zero_grad()
            
        y_hat = pfno(x)
        print('y_hat: ', y_hat.shape)
        print('y: ', y.shape)

        loss = criterion(y_hat, y)

        if P_root.active:
            train_loss += loss.item()
            n_train_batch += 1

        loss.backward()
        if optimizer is not None:
            optimizer.step()

    if P_root.active:
        train_accs.append(train_loss/n_train_batch)
    
    P_x._comm.Barrier()
    
    # Loop over validation data
    pfno.eval()
    valid_loss = 0
    n_valid_batch = 0

    for j, (x, y) in enumerate(valid_loader):
        with torch.no_grad():
            x = x.to(P_x.device)
            y = y.to(P_x.device)

            y_hat = pfno(x)
            loss = criterion(y_hat, y)
            if P_root.active:
                valid_loss += loss.item()
                n_valid_batch += 1

    t1 = time.time()
    if P_root.active:
        print(f'epoch = {i}, train loss = {train_loss/n_train_batch:08f}, valid loss = {train_loss/n_train_batch:08f}, time = {t1 - t0:04f}')
        valid_accs.append(valid_loss/n_valid_batch)

    if (i+1) % checkpoint_interval == 0:

        # Save loss
        if P_root.active:
            lossname = 'loss_epoch_' + str(i) + '.h5'
            fid = h5py.File(os.path.join(out_dir, lossname), 'w')
            fid.create_dataset('train_loss', data=train_accs)
            fid.create_dataset('valid_loss', data=valid_accs)
            fid.close()

        # Save snapshot
        model_path = os.path.join(out_dir, f'model_snapshot_{i:04d}_{P_x.rank:04d}.pt')
        torch.save({
            'epoch': i,
            'model_state_dict': pfno.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 
            model_path)

# Save after training
model_path = os.path.join(out_dir, f'model_sleipner_{P_x.rank:04d}.pt')
torch.save(pfno.state_dict(), model_path)
print(f'rank = {P_x.rank}, saved model after final iteration: {model_path}')

if P_root.active:
    print('training finished.')
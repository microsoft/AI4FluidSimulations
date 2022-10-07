#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import sys 
sys.path.append('..')
from distdl.backend import BackendProtocol, FrontEndProtocol, ModelProtocol, init_distdl
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError, R2Score
from distdl.backends.common.partition import MPIPartition
from distdl.nn.repartition import Repartition
import distdl, torch, os, time, h5py
import numpy as np
from mpi4py import MPI
import azure
from pfno import ParallelFNO4d
from dataset import SleipnerParallel
import matplotlib.pyplot as plt

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

# Collectors
collect_x = Repartition(P_x, P_root)
collect_y = Repartition(P_x, P_root)
collect_y_ = Repartition(P_x, P_root)

# Reproducibility
torch.manual_seed(P_x.rank + 123)
np.random.seed(P_x.rank + 123)

# Data dimensions
nb = 1
shape = (262, 118, 64, 86)    # X Y Z T
num_train = 1576
num_test = 24

# Network dimensions
channel_in = 4
channel_hidden = 10
channel_out = 1
num_k = (16, 16, 12, 10)

# Data store
account_url = "https://" + os.environ['BLOB_ACCOUNT'] + ".blob.core.windows.net"
container = os.environ['BLOB_CONTAINER']
data_path = '/data'

client = azure.storage.blob.ContainerClient(
    account_url=account_url,
    container_name=container,
    credential=os.environ['BLOB_KEY']
    )

# Training dataset
test_idx = torch.linspace(num_train, num_train + num_test, num_test, dtype=torch.int32).long()
test_data = SleipnerParallel(P_x, test_idx, client, container, data_path, shape, savepath=os.environ['DATA_DIR'], filename='sleipner')

# Dataloaders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=nb, shuffle=False)
P_world._comm.Barrier()


# FNO
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

criterion = distdl.nn.DistributedMSELoss(P_x).to(P_x.device)

# Load trained network
out_dir = os.environ['MODEL_DIR']
model_path = os.path.join(out_dir, f'model_sleipner_{P_x.rank:04d}.pt')
pfno.load_state_dict(torch.load(model_path))
pfno = pfno.to(P_x.device)
pfno.eval()

# Get sample
if P_world.rank == 0:
    mse = MeanSquaredError().to(P_x.device)
    mape = MeanAbsolutePercentageError().to(P_x.device)
    r2 = R2Score().to(P_x.device)

    mse_scores = torch.zeros(len(test_loader))
    mape_scores = torch.zeros(len(test_loader))
    r2_scores = torch.zeros(len(test_loader))

for i, (x, y) in enumerate(test_loader):

    if P_world.rank == 0:
        print("Process sample: ", i, "\n")

    x = x.to(P_x.device)
    y = y.to(P_x.device)

    # Predict
    with torch.no_grad():
        t0 = time.time()
        y_  = pfno(x)
        t1 = time.time()
        print(t1 - t0)

    # Collect on root
    x = collect_x(x)
    y = collect_y(y)
    y_ = collect_y_(y_)

    if P_world.rank == 0:
        mse_scores[i] = mse(y.reshape(-1), y_.reshape(-1))
        mape_scores[i] = mape(y.reshape(-1), y_.reshape(-1))
        r2_scores[i] = r2(y.reshape(-1), y_.reshape(-1))

if P_world.rank == 0:
    torch.save(mse_scores, 'mse_scores.pt')
    torch.save(mape_scores, 'mape_scores.pt')
    torch.save(r2_scores, 'r2_scores.pt')

    print("R2: ", r2_scores.mean())
    print("MSE: ", mse_scores.mean() * 1e3)
    print("MAPE: ", mape_scores.mean() * 1e3)

# Save result
if P_root.active:
    fid = h5py.File(os.path.join(out_dir, 'fno_sample.h5'), 'w')
    fid.create_dataset('x', data=x.detach().cpu())
    fid.create_dataset('y_', data=y_.detach().cpu())
    fid.create_dataset('y', data=y.detach().cpu())
    print("Saved data sample!")

    plt.figure(figsize=(8,4))
    plt.subplot(1,3,1)
    plt.imshow(x[0,-1,205,:,:,0].detach().cpu())
    plt.title("Well location")
    plt.subplot(1,3,2)
    plt.imshow(y[0,0,205,:,:,-1].detach().cpu())
    plt.title("Target")
    plt.subplot(1,3,3)
    plt.imshow(y_[0,0,205,:,:,-1].detach().cpu())
    plt.title("Prediction")
    plt.savefig("comparison.png")
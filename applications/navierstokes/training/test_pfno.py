import distdl, torch, os, time, h5py
import numpy as np
from mpi4py import MPI
from pfno3d import ParallelFNO3d
from distdl.backends.mpi.partition import MPIPartition
from dataset import WaterlilySerial
from distdl.nn.repartition import Repartition
import matplotlib.pyplot as plt

# Init MPI
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()
n = P_world.shape[0]

# Master worker partition with 6 dimensions ( N C X Y T )
root_shape = (1, 1, 1, 1, 1)
P_root_base = P_world.create_partition_inclusive([0])
P_root = P_root_base.create_cartesian_topology_partition(root_shape)

# Distributed paritions
feat_workers = np.arange(0, n)
P_feat_base = P_world.create_partition_inclusive(feat_workers)
P_x = P_feat_base.create_cartesian_topology_partition((1,1,1,n,1))
P_y = P_feat_base.create_cartesian_topology_partition((1,1,n,1,1))

# Collectors
collect_x = Repartition(P_x, P_root)
collect_y = Repartition(P_x, P_root)
collect_y_ = Repartition(P_x, P_root)

# Cuda
device = torch.device(f'cuda:{P_x.rank}')

# Reproducibility
torch.manual_seed(P_x.rank + 123)
np.random.seed(P_x.rank + 123)

# Data dimensions
nb = 1
shape = (130, 130, 50)    # X Y Z T
num_train = 1800
num_test = 200

# Network dimensions
channel_in = 6
channel_hidden = 16
channel_out = 1
num_k = (16, 16, 8)

# Test dataset
path = '/scratch/pwitte/waterlily/data'
basename = 'cylinder_2d'
test_idx = torch.linspace(1 + num_train, num_test + num_train, num_test, dtype=torch.int32).long()
test_data = WaterlilySerial(test_idx, path, basename, normalize=True, clip=.3, target='vorticity')

# Dataloaders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=nb, shuffle=False)
P_world._comm.Barrier()

# FNO
pfno = ParallelFNO3d(
    P_world, 
    P_root,
    P_x,
    P_y,
    channel_in,
    channel_hidden,
    channel_out,
    num_k
)

# Load trained network
out_dir = '/scratch/pwitte/waterlily/model'
#model_path = os.path.join(out_dir, f'model_{P_x.rank:04d}.pt')
model_path = os.path.join(out_dir, 'model_0049_0000.pt')
pfno.load_state_dict(torch.load(model_path))
pfno = pfno.to(device)
pfno.train()

# Get sample
data_iterator = iter(test_loader)
x, y = next(data_iterator)
x, y = next(data_iterator)
x, y = next(data_iterator)

x = x.to(device)
y = y.to(device)

# Predict
with torch.no_grad():
    y_  = pfno(x)

# Collect on root
x = collect_x(x)
y = collect_y(y)
y_ = collect_y_(y_)

# Save result
if P_root.active:
    fid = h5py.File(os.path.join(out_dir, 'fno_sample.h5'), 'w')
    fid.create_dataset('x', data=x.detach().cpu())
    fid.create_dataset('y_', data=y_.detach().cpu())
    fid.create_dataset('y', data=y.detach().cpu())
    print("Saved data sample!")

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x[0,0,:,:,0].detach().cpu())
    plt.title("Input (channel 0)")
    plt.subplot(1,3,2)
    plt.imshow(y[0,0,:,:,-1].detach().cpu())
    plt.title("Target")
    plt.subplot(1,3,3)
    plt.imshow(y_[0,0,:,:,-1].detach().cpu())
    plt.title("Prediction")
    plt.savefig("comparison.png")

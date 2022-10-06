#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import sys, os
import torch, zarr
from dataset import SleipnerSerial
import azure.storage.blob
import matplotlib.pyplot as plt

# fix the seed
torch.manual_seed(0)
mode = 'fno'

batch_size = 1
restart = False
container = 'sleipner'
data_path = 'data'

# Computational grid
nx = 263
ny = 118
nz = 64
nt = 86

# Az storage client
client = azure.storage.blob.ContainerClient(
    account_url="https://pwittesleipner.blob.core.windows.net",
    container_name=container,
    credential=os.environ['SECRET_KEY']
    )

# Number of samples to use in training
num_train_data = 1600
train_idx = torch.linspace(1, num_train_data, num_train_data, dtype=torch.int32).long()

# Sleipner dataset
train_data = SleipnerSerial(train_idx, client, container, data_path, (nx, ny, nz), normalize=False, savepath='/scratch/pwitte/sleipner', filename="sample", target="saturation")

# Plot sample
x, y = train_data[0]
plt.subplot(2,1,1)
plt.imshow(y[0, 205,:,:,-1])
plt.subplot(2,1,2)
plt.imshow(y[0,:,:,38,-1])
plt.tight_layout()
plt.savefig("y.png")
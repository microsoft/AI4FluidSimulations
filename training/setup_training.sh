#!/bin/bash

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

CUDA_VERSION=$1
CUPY_VERSION=${CUDA_VERSION//./}

# DistDl
pip3 install --no-cache-dir git+https://github.com/philippwitte/distdl.git@nccl

# Dependencies
pip3 install --no-cache-dir azure-common azure-storage-blob h5py zarr mpi4py matplotlib torchmetrics

# Cupy + nccl support
pip3 install --no-cache-dir cupy-cuda$CUPY_VERSION --pre -f https://pip.cupy.dev/pre
python3 -m cupyx.tools.install_library --cuda $CUDA_VERSION --library nccl
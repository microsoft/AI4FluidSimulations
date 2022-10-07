#!/bin/bash

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Install Julia dependencies
julia -e 'using Pkg; Pkg.add(["AzureClusterlessHPC", "PyCall", "WaterLily", "HDF5", "JLD"])'

# Install Python dependencies
pip3 install --no-cache-dir setuptools azure-batch==9.0.0 azure-common azure-storage-blob==1.3.1 azure-storage-queue==1.4.0
# Model-parallel FNO


## Run with docker

The quickest way to get started is with the pre-built docker image. Simply run the following commands to start an interactive docker session:

```
# Go to the training directory
cd ~/AI4FluidSimulations/training

# Start the container interactively
docker run --gpus all \
    -v $(pwd):/workspace/home \
    -e OMPI_ALLOW_RUN_AS_ROOT="1" \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
    -it philippwitte/ai4fluidsimulations-training:v1.0
```

This commands mounts the `AI4FluidSimulations` directory into the container. You can run the example in `training/example_pfno.py` via:

```
# Run example on 4 GPUs
mpiexec -n 4 python3 example_pfno.py
```

## Run on host

To run the examples directly on your host, install the dependencies using the `setup_training.sh` script. Pass the CUDA version of your system as an argument (set `x` accordingly):

```
# CUDA version 11.x
bash setup_training.sh 11.x
```

If everything has installed correctly, you can run the example script:

```
# Run example on 4 GPUs
mpiexec -n 4 python3 example_pfno.py
```

## Navier-Stokes and CO2 example

The Navier-Stokes and Sleipner examples read training data from blob storage and (optionally) cache it on a hard drive. Before running the training script, set the following environment variables that point to your blob storage account with the training data. The variables `DATA_DIR` and `MODEL_DIR` are the output directories for the training data and trained network respectively.

```
# Blob storage
export BLOB_ACCOUNT=""
export BLOB_KEY=""
export BLOB_CONTAINER=""

# Local directories
export DATA_DIR=""
export MODEL_DIR=""
```

Run the training or testing scripts via:

```
# Go to navier_stokes or sleipner directory
cd navier_stokes

# Training
mpiexec -n 8 python3 train_pfno.py

# Testing
mpiexec -n 8 python3 test_pfno.py
```
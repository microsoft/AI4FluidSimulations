# Model-parallel FNO

## Prerequisits

## Build docker (optional)

```
cd ~/AI4FluidSimulations/docker
docker build -f Dockerfile.Training -t ai4fluidsimulations-training:v1.0 .
```

## Run with docker

To run the docker container interactively, we start the container and mount the `training` directory with the code into the container:

```
# Go to the training directory
cd ~/AI4FluidSimulations/training

# Start the container interactively
docker run --gpus all \
    -v $(pwd):/workspace/home \
    -e OMPI_ALLOW_RUN_AS_ROOT="1" \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
    -it philippwitte/ai4fluidsimulations-train:v1.0
```

## Run on host

To run the examples directly on your host, install the dependencies using the `setup_training.sh` script. Pass the CUDA version of your system as an argument:

```
# Install dependencies for CUDA 11.6
bash setup_training.sh 11.6
```
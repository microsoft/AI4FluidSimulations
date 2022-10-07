# AI4IndustrySimulations

## Overview

**AI for Industry Simulations** is a project for training large-scale surrogate models for solving partial differential equations (PDEs) with deep learning. We target large-scale three-dimensional applications as common in industrial applications such as reservoir simulation. The current repository contains two example applications:

- Simulating two-phase CO2 flow in porous media.

- Solving the 3D Navier Stokes to simulate flow around a sphere.

For each example, we provide the code to simulate the training data and to train a neural surrogate model using a [model-parallel implementation of Fourier Neural Operators](https://arxiv.org/abs/2204.01205). We train our deep surrogate model using supervised training, so simulating the training data by solving the underlying PDE for different inputs is the first step of the workflow. For industry-sized applications, training data simulation is time consuming, as we need to solve 3D PDEs for a large number of samples. We provide examples for simulating training data in parallel on Azure using the AzureClusterlessHPC package and for storing the data in Azure's cloud object store (Blob Storage).

For training, we use a model-parallel version of [Fourier Neural Operators](https://arxiv.org/pdf/2010.08895.pdf). The model-parallel FNO uses domain decomposition, which enables a higher level of concurrently than model sharding or pipeline parallelism. The model-parallel FNO is based on distributed programming with [DistDL](https://github.com/distdl/distdl), a Python package with distributed communication primitives for implementing model-parallel neural networks.


## Quickstart for parallel FNO training

Clone this repository:

```
https://github.com/microsoft/AI4FluidSimulations
```

Go to the training directory and pull our docker image for training:

```
# Go to examples directory
cd AI4FluidSimulations/training

# Start docker container
docker run --gpus all \
    -v $(pwd):/workspace/home \
    -e OMPI_ALLOW_RUN_AS_ROOT="1" \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
    -it philippwitte/ai4fluidsimulations-training:v1.0
```

Run the example script (e.g., on 4 GPUs):

```
mpiexec -n 4 python3 example_pfno.py
```

## Data simulation on Azure

Follow the instructions [here](https://github.com/microsoft/AI4FluidSimulations/tree/main/simulation). An Azure subscription is required for data generation.

## Credits

This repository is developed and maintained by the [Microsoft Research for Industry](https://www.microsoft.com/en-us/research/group/research-for-industry/) (RFI) team.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
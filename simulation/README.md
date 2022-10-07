# Training data simulation

This directory contains the code to simulate the training data for the Sleipner CO2 example and the 3D Navier-Stokes equation. Generating the datasets requires an Azure subscription and the installation of [AzureClusterlessHPC.jl](https://github.com/microsoft/AzureClusterlessHPC.jl). 


## Prerequisits

- Julia version 1.7 or above.

- An Azure subscription.

- An Azure Batch account with Azure Active Directory (AAD) authentication and an Azure Blob storage account. You can follow the instructions from the AzureClusterlessHPC.jl [documentation](https://microsoft.github.io/AzureClusterlessHPC.jl/installation/). 

- Populate the `credentials.json` file with the information for you batch and storage accounts.

- Install the dependencies via `bash setup_simulation.sh`


## Docker (optional)

The Sleipner and Navier Stokes example each have their own docker image with the required software dependencies. You do not need to build the docker images yourself, as Azure Batch will pull existing images from a public image repository.

If you wish to re-build the container images (e.g., to update dependencies or make modifications), you can use the Dockerfiles in `AI4FluidSimulation/docker`:

```
cd ~/AI4FluidSimulations/docker

# Navier Stokes example
docker build -f Dockerfile.NavierStokes -t redwood-navierstokes:v1.0 .

# Sleipner example
docker build -f Dockerfile.CO2Flow -t redwood-co2flow:v1.0 .
```

## Navier-Stokes example

Running the example requires an Azure Storage account to which the simulated training data will be written. This blob account can be different than the one that AzureClusterlessHPC.jl is using. Before running the example, create a Blob storage account and export environment variables for the account name, container name and secret key:

```
export BLOB_ACCOUNT="blob_storage_account_name"
export BLOB_CONTAINER="blob_storage_container_name"
export BLOB_KEY="blob_storage_secret_key"
```

Make sure that the container exists in your storage account prior to running the example, as the container will not be created automatically.

To start the training data simulation, run:

```
cd ~/AI4FluidSimulations/simulation/navier_stokes
julia flow_cylinder_3d.jl
```

**Note**: The default number of VMs in `parameters.json` is 4 VMs. To simulate the full dataset (3,200 data samples), increase the number of VMs and set the variable `num_train=3200` (in flow_cylinder.jl). The runtime per sample is approximately 15 minutes.


## CO2 flow example

As for the Navier Stokes example, export the name, container and credentials for your storage account:

```
export BLOB_ACCOUNT="blob_storage_account_name"
export BLOB_CONTAINER="blob_storage_container_name"
export BLOB_KEY="blob_storage_secret_key"
```

Start the script via:

```
cd ~/AI4FluidSimulations/simulation/sleipner
julia run_opm_batch.jl
```

**Note**: The example uses 4 VMs and simulates 4 training data examples. The runtime per example is between 4 and 12 hours. To increase the number of simulated training examples, adjust the pool size and the `num_train` variable in the Julia script accordingly.
# Training data simulation

This directory contains the code to simulate the training data for the Sleipner CO2 example and the 3D Navier-Stokes equation. Generating the datasets requires an Azure subscription and the installation of [AzureClusterlessHPC.jl](https://github.com/microsoft/AzureClusterlessHPC.jl). 

## Prerequisits

- Julia version 1.7 or above.

- An Azure subscription.

- Installation of the Julia package [AzureClusterlessHPC.jl](https://github.com/microsoft/AzureClusterlessHPC.jl).

- An Azure Batch account with Azure Active Directory (AAD) authentication and an Azure Blob storage account. You can follow the instructions from the AzureClusterlessHPC.jl [documentation](https://microsoft.github.io/AzureClusterlessHPC.jl/installation/). 

- Populate the `credentials.json` file with the information for you batch and storage accounts.

## Docker images for data generation


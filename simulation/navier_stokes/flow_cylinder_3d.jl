#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

#######################################################################################################################
# Cloud setup

ENV["CREDENTIALS"] = joinpath(pwd(), "../credentials.json")
ENV["PARAMETERS"] = joinpath(pwd(), "parameters.json")

using AzureClusterlessHPC, PyCall
batch_clear()

# Create pool
create_pool()

######################################################################################################################
# Simulator

@batchdef using WaterLily, PyCall, Random

@batchdef function circle(id, url, container, credential, n, radius, ν, stop, center)

    # Simulator
    body = AutoBody((x, t)->WaterLily.norm2(x .- center) - radius)
    sim = Simulation((n, n, n), [1., 0.5, 0.], radius; ν, body)
    
    # Solve flow
    t = range(0., stop; step=1)
    nt = length(t)
    ω_hist = zeros(n, n, n, length(t))
    p_hist = zeros(n, n, n, length(t))

    for (i, tᵢ) in enumerate(t)
        print("Simulate time step: ", i)
        @time sim_step!(sim, tᵢ)
        @inside sim.flow.σ[I] =  WaterLily.curl(3, I, sim.flow.u)
        ω_hist[:, :, :, i] = sim.flow.σ
        p_hist[:, :, :, i] = sim.flow.p
    end

    # Collect in- and outputs
    model = cat(sim.flow.μ₀, reshape(sim.flow.μ₁, n, n, n, :), dims=4)

    # Zarr client
    blob = pyimport("azure.storage.blob")
    client = blob.ContainerClient(
        account_url=url,
        container_name=container,
        credential=credential
    )

    # Store data in Azure blob storage
    zarr = pyimport("zarr")
    store = zarr.ABSStore(container=container, prefix="data", client=client)  
    root = zarr.group(store=store, overwrite=false)
    root.array("model_" * string(id), model, chunks=(32, 32, 32, 32), overwrite=true)
    root.array("vorticity_" * string(id), ω_hist, chunks=(32, 32, 32, 32), overwrite=true)
    root.array("pressure_" * string(id), p_hist, chunks=(32, 32, 32, 32), overwrite=true)
end

#######################################################################################################################
# Generate data

# Create simulator
n = 2^7 + 2
radius = 8
Re = 200
ν = 8 / Re
stop = 64.
num_train = 4 # 3,200 for the full dataset

# Data store info
account = ENV["BLOB_ACCOUNT"]
container = ENV["BLOB_CONTAINER"]
credential = ENV["BLOB_KEY"]
url = "https://" * account * ".blob.core.windows.net"

# Create centers
centers = vec(CartesianIndices((16:Int(n/2)-16, 16:n-16, 16:n-16)))
idx = Array(1:length(centers))

# Draw random center locations and store in list
shuffle!(idx)
center_list = []
for i=1:num_train
    push!(center_list, Array([centers[idx[i]].I...]))
end

# Run jobs
bctrl = @batchexec pmap(i -> circle(i, url, container, credential, n, radius, ν, stop, center_list[i]), 1:num_train)
wait_for_tasks_to_complete(bctrl; timeout=9999, task_timeout=60, num_restart=9)

# Delete resource
destroy!(bctrl)

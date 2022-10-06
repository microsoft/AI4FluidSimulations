#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# Simulate CO2 movement using OPM and Azure batch_clear
# Philipp Witte
# August 2022
# Microsoft
#

#######################################################################################################################
# Batch pool

ENV["CREDENTIALS"] = joinpath(pwd(), "../credentials.json")
ENV["PARAMETERS"] = joinpath(pwd(), "parameters.json")

using AzureClusterlessHPC, PyCall, HDF5, JLD
batch_clear()

# Create pool
create_pool()


#######################################################################################################################
# Run OPM

# Load packages
@batchdef using PyCall, Random

# Include files
@batchdef begin
    fileinclude("SLEIPNER_ORG.DATA")
    fileinclude("DX.INC")
    fileinclude("DY.INC")
    fileinclude("DZ.INC")
    fileinclude("SCHEDULE.INC")
    fileinclude("SUMMARY.INC")
    fileinclude("FIPNUM.INC")
    fileinclude("TOPS.INC")
    fileinclude("gen_model_sleipner.py")
end

# Include current path in Python environment
@batchdef pushfirst!(PyVector(pyimport("sys")."path"), pwd())

# Run OPM
@batchdef function run_opm(isim, filename, shape, nbpml, opm_cmd, url, container, credential)

    # Zarr client
    blob = pyimport("azure.storage.blob")
    client = blob.ContainerClient(
        account_url=url,
        container_name=container,
        credential=credential
    )

    # Run model generation script
    model_generator = pyimport("gen_model_sleipner")
    model_generator.gen_model()
    well = model_generator.gen_sample() # Y X

    # Run OPM
    ENV["OMP_NUM_THREADS"] = 1
    run(opm_cmd)

    # Read grid
    grid = pyimport("ecl.grid")
    ecl = pyimport("ecl.eclfile")
    grid = grid.EclGrid(join([filename, ".EGRID"]))

    # Read snapshots & models
    rst_file = ecl.EclRestartFile(grid, join([filename, ".UNRST"]))

    # Read pressure & saturation
    nt = length(get(rst_file, "PRESSURE"))
    pressure = zeros(Float32, (nt, shape...))
    saturation = zeros(Float32, (nt, shape...))
    shape_pml = (shape[1] + 2*nbpml, shape[2] + 2*nbpml, shape[3])
    for j=1:nt
        pressure[j, :, :, :] = reshape(get(rst_file, "PRESSURE")[j].numpy_view(), shape_pml)[nbpml+1:end-nbpml, nbpml+1:end-nbpml, :]
        saturation[j, :, :, :] = reshape(get(rst_file, "SGAS")[j].numpy_view(), shape_pml)[nbpml+1:end-nbpml, nbpml+1:end-nbpml, :]
    end

    # Reshuffle dimensions to (Z Y X T)
    pressure = permutedims(pressure, (4,3,2,1))
    saturation = permutedims(saturation, (4,3,2,1))

    # Write results to blob
    zarr = pyimport("zarr")
    store = zarr.ABSStore(container=container, prefix="data", client=client)  
    root = zarr.group(store=store, overwrite=false)
    root.array("well_" * string(isim), well[nbpml+1:end-nbpml, nbpml+1:end-nbpml], chunks=(32, 32), overwrite=true)
    root.array("pressure_" * string(isim), pressure, chunks=(32, 32, 32, 32), overwrite=true)
    root.array("saturation_" * string(isim), saturation, chunks=(32, 32, 32, 32), overwrite=true)
end


#######################################################################################################################

# Required arguments
filename = "SLEIPNER_ORG"
shape = (64, 118, 263)    # nx, ny, nz
nbpml = 4
opm_cmd = `mpiexec -n 4 --allow-run-as-root flow $filename.DATA`
num_train = 4   # 1600 for the full dataset

# Populate with Azure credentials
account = ENV["BLOB_ACCOUNT"]
container = ENV["BLOB_CONTAINER"]
credential = ENV["BLOB_KEY"]
url = "https://" * account * ".blob.core.windows.net"


bctrl = @batchexec pmap(isim -> run_opm(isim, filename, shape, nbpml, opm_cmd, url, container, credential), 1:num_train)
wait_for_tasks_to_complete(bctrl; timeout=9999, task_timeout=720, num_restart=4)

# Get pool start time and job stats
pool = AzureClusterlessHPC.__clients__[1]["batch_client"].pool
start_time = pool.list().next().creation_time
job_stats = get_job_stats(bctrl)

# Save stats
save("timings.jld", "job_stats", job_stats, "start_time", start_time)

# Delete pool and temporary blob resources
destroy!(bctrl)
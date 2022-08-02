using PyCall, Random, HDF5, PyPlot

# Include current path in Python environment
pushfirst!(PyVector(pyimport("sys")."path"), pwd())

# Read grid
grid = pyimport("ecl.grid")
ecl = pyimport("ecl.eclfile")
grid = grid.EclGrid("SLEIPNER_ORG.EGRID")

init_file = ecl.EclInitFile(grid, "SLEIPNER_ORG.INIT")
keys = init_file.keys()

# Blob client
blob = pyimport("azure.storage.blob")
url = "https://myblobaccount.blob.core.windows.net"
container = "mycontainer"
credential = "mysecretkey"
client = blob.ContainerClient(
    account_url=url,
    container_name=container,
    credential=credential
)

# Read parameters
nbpml = 4
shape = (64 + 2*nbpml, 118 + 2*nbpml, 263)    # X Y Z
depth = reshape(get(init_file, "DEPTH")[1].numpy_view(), shape)[nbpml+1:end-nbpml, nbpml+1:end-nbpml, :];
permx = reshape(get(init_file, "PERMX")[1].numpy_view(), shape)[nbpml+1:end-nbpml, nbpml+1:end-nbpml, :];
permz = reshape(get(init_file, "PERMZ")[1].numpy_view(), shape)[nbpml+1:end-nbpml, nbpml+1:end-nbpml, :];

# X Y Z  -> Z Y X
permx = permutedims(permx, (3,2,1))
permz = permutedims(permz, (3,2,1))
depth = permutedims(depth, (3,2,1))

# Write results to blob
zarr = pyimport("zarr")
store = zarr.ABSStore(container=container, prefix="data", client=client)  
root = zarr.group(store=store, overwrite=false)
root.array("permx" , permx, chunks=(32, 32, 32), overwrite=true)
root.array("permz" , permz, chunks=(32, 32, 32), overwrite=true)
root.array("depth" , depth, chunks=(32, 32, 32), overwrite=true)

# X Y Z
subplot(3,1,1)
imshow(permx[:,:,32]); colorbar(); title("PERMX")
subplot(3,1,2)
imshow(permz[:,:,32]); colorbar(); title("PERMZ")
subplot(3,1,3)
imshow(depth[:,:,32]); colorbar(); title("DEPTH")
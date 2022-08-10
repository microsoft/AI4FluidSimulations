from ecl.grid import EclGrid
import numpy as np
import matplotlib.pyplot as plt
import h5py, torch

# Load grid ecl file
sleipner = EclGrid.load_from_grdecl('Sleipner_Reference_Model_cleaned.grdecl')

nx = sleipner.get_nx()
ny = sleipner.get_ny()
nz = sleipner.get_nz()
nbpml = 4

# Read TOP
tops = np.ones((nx, ny))
for i in range(nx):
    for j in range(ny):
        tops[i,j] = sleipner.top(i,j)

# Grid spacing
dx = np.ones((nx + 2*nbpml, ny + 2*nbpml, nz)) * 50
dy = np.ones((nx + 2*nbpml, ny + 2*nbpml, nz)) * 50
dz = np.ones((nx, ny, nz))

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            dz[i,j,k] = sleipner.get_cell_dims(ijk=(i,j,k))[2]

# Extend and bcs
tops = np.pad(tops, ((nbpml, nbpml), (nbpml, nbpml)), mode='reflect')
dz = np.pad(dz, ((nbpml, nbpml), (nbpml, nbpml), (0, 0)), mode='reflect')

# Increase grid spacing at boundaries
dxy_dec = 50*3**np.linspace(0,nbpml-1,nbpml)[::-1]
print("Size of PML: ", dxy_dec.sum()/1e3)

dx[:, :, :] = dxy_dec[0]
dy[:, :, :] = dxy_dec[0]
for i in range(1,nbpml):
    dx[i:-i, i:-i, :] = dxy_dec[i]
    dy[i:-i, i:-i, :] = dxy_dec[i]

# Transpose for OPM to (Z Y X)
dx = dx.transpose(2,1,0)
dy = dy.transpose(2,1,0)
dz = dz.transpose(2,1,0)
tops = tops.transpose(1,0)

# Save files
np.savetxt('TOPS.INC', tops.flatten().reshape( (-1, 8)), delimiter='\t', header='TOPS', comments='', fmt='%.6f', footer='/')
np.savetxt('DX.INC', dx.flatten().reshape( (-1, 8)), delimiter='\t', header='DX', comments='', fmt='%.6f', footer='/')
np.savetxt('DY.INC', dy.flatten().reshape( (-1, 8)), delimiter='\t', header='DY', comments='', fmt='%.6f', footer='/')
np.savetxt('DZ.INC', dz.flatten().reshape( (-1, 8)), delimiter='\t', header='DZ', comments='', fmt='%.6f', footer='/')

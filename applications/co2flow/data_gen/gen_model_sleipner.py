import numpy as np
import os


def draw_well_loc(wells):
    ny, nx = wells.shape
    while True:
        ix = np.random.randint(16,nx-16)
        iy = np.random.randint(16,ny-16)
        if 1 not in wells[iy-1:iy+2, ix-1:ix+2]:
            wells[iy-1:iy+2, ix-1:ix+2] = 1
            break
    return wells, ix, iy


def gen_sample():
    
    # Well loc
    minwell = 1
    maxwell = 4
    nwell = np.random.randint(minwell, maxwell+1)
    wellmap = np.zeros((126, 72))
    wellx = []
    welly = []
    welltxt = open('WELSPECS.txt', 'w')
    welltxt.write('WELSPECS\n')
    comptxt = open('COMPDAT.txt', 'w')
    comptxt.write('COMPDAT\n')
    wcontxt = open('WCONINJE.txt', 'w')
    wcontxt.write('WCONINJE\n')
    iz = 214
    nperf = 8

    for i in range(nwell):
        wellmap, ix, iy = draw_well_loc(wellmap)
        wellx.append(ix)
        welly.append(iy)
        name = 'Inj' + str(i)
        welltxt.write('{} I {} {} 1000 GAS /\n'.format(name, ix, iy))
        comptxt.write('{} {} {} {} {} OPEN -1 8.5e+02 2.0e-01 -1.0 0 1* Y -1.0 /\n'.format(name, ix, iy, iz, iz+nperf))
        wcontxt.write('{} GAS OPEN RATE {} 3* 0 /\n'.format(name, 1e6/nwell))

    welltxt.close()
    comptxt.close()
    wcontxt.close()

    return wellmap


def gen_model():
    # Dimensions
    nx = 64
    ny = 118
    nz = 263
    nbpml = 4

    # Permeability
    permx = np.zeros((nx + 2*nbpml, ny + 2*nbpml, nz))
    permx[:,:,0:10] = 0.00140825
    permx[:,:,10:31] = 1300.53
    permx[:,:,31:47] = 0.00208193
    permx[:,:,47:63] = 1746.74
    permx[:,:,63:66] = 0.00150106
    permx[:,:,66:82] = 3364.88
    permx[:,:,82:85] = 0.00245556
    permx[:,:,85:102] = 2655.74
    permx[:,:,102:105] = 0.00246601
    permx[:,:,105:120] = 1712.74
    permx[:,:,120:123] = 0.000959924
    permx[:,:,123:146] = 2122.88
    permx[:,:,146:149] = 0.000968739
    permx[:,:,149:164] = 2084.37
    permx[:,:,164:167] = 0.00125205
    permx[:,:,167:202] = 3044.56
    permx[:,:,202:205] = 0.00170393
    permx[:,:,205:263] = 3483.09

    # PERMZ
    permz = np.copy(permx)
    permz[29:31,38:40,31:47] = 0.000316709
    permz[29:31,38:40,63:66] = 5.46816
    permz[29:31,38:40,82:85] = 46.2126
    permz[29:31,38:40,102:105] = 0.104632
    permz[29:31,38:40,120:123] = 0.0122186
    permz[29:31,38:40,146:149] = 949.193
    permz[29:31,38:40,164:167] = 0.0665459
    permz[29:31,38:40,202:205] = 0.0627328
    permz[39:42,54:58,120:123] = 70.7322
    permz[22:27,14:23,164:167] = 2495.31

    # Porosity
    poro = np.ones((nx + 2*nbpml, ny + 2*nbpml, nz)) * 0.34

    # Reshape for OPM (X Y Z -> Z Y X)
    permx = permx.transpose(2,1,0) / 80    # 60?
    permz = permz.transpose(2,1,0) / 80    # 60?
    poro = poro.transpose(2,1,0)

    # FIPNUM
    fip = []
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 10) + '*1')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 21) + '*2')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 16) + '*3')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 16) + '*4')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 3) + '*5')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 16) + '*6')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 3) + '*7')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 17) + '*8')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 3) + '*9')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 15) + '*10')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 3) + '*11')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 23) + '*12')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 3) + '*13')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 15) + '*14')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 3) + '*15')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 35) + '*16')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 3) + '*17')
    fip.append(str((nx + 2*nbpml) * (ny + 2*nbpml) * 58) + '*18')
    fip = np.array(fip)

    # Save
    np.savetxt('PERMX.INC', permx.flatten().reshape( (-1, 8)), delimiter='\t', header='PERMX', comments='', fmt='%.6f', footer='/')
    np.savetxt('PERMZ.INC', permz.flatten().reshape( (-1, 8)), delimiter='\t', header='PERMZ', comments='', fmt='%.6f', footer='/')
    np.savetxt('PORO.INC', poro.flatten().reshape( (-1, 8)), delimiter='\t', header='PORO', comments='', fmt='%.6f', footer='/')
    np.savetxt('FIPNUM.INC', fip.flatten(), delimiter='\t', header='FIPNUM', comments='', fmt='%s', footer='/')


if __name__ == '__main__':
    gen_model()
    gen_sample()
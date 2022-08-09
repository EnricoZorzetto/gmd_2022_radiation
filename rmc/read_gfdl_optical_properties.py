
# read single-time-step optical properties provided by Ray

################################################################################

###                          RAD DATA

################################################################################


import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
# import demfuncs as dem
import photon_mc_numba_fractional as ph
import  photon_mc_atmosphere as atm
import json

from itertools import cycle

ph.matplotlib_update_settings()

mdfile = 'exp/laptop_test_gfdl_singletimestep.json'
print('mdfile = {}'.format(mdfile))
metadata = json.load(open(mdfile, 'r'))
# dfatm = atm.init_atmosphere_gfdl_singletimestep(metadata)
#
# plt.figure()
# plt.plot(dfatm['zz'])
# plt.plot(dfatm['zz_hydrostatic'])
# plt.show()



# dfatm['k_ext_tot']

# pressm = ds['pressm']
# pressm[0,:,0,0]


datadir = "/Users/ez6263/Documents/rmc_datasets/"
# datadir = "/lustre/f2/dev/Enrico.Zorzetto/dem_datasets/gmd_2021_dems/"
# datadir = os.path.join('//', 'home', 'enrico', 'Documents',
#                        'dem_datasets')

# outfigdir = os.path.join(datadir, 'figures_optical_gfdl')
# outfigdir = os.path.join(datadir, 'am4-single-day-optics.nc')
outfigdir = os.path.join(datadir, 'figures_optical_gfdl')

if not os.path.exists(outfigdir):
    os.makedirs(outfigdir)

print(os.listdir(datadir))

filename = os.path.join(datadir, 'single-time-step.nc')
# filename = os.path.join(datadir, 'atmos_single_day.nc')

# with open(filename, 'r') as f:
ds = Dataset(filename, 'r')
for keyi in list(ds.variables):
    print('__{}__'.format(keyi))
    # print(ds[keyi])


#------------------------- GET STUDY SITE POSITION ----------------------------#
clat = (metadata['maxlat'] + metadata['minlat']) / 2.0
clon = (metadata['maxlon'] + metadata['minlon']) / 2.0
xt = np.mean(ds['grid_xt_bnds'][:], axis=1)  # grid cell lon centers 0-360
yt = np.mean(ds['grid_yt_bnds'][:],
             axis=1)  # grid cell lat centers -90 - 90
indx = np.argmin(np.abs(xt - clon))
indy = np.argmin(np.abs(yt - clat))
#------------------------------------------------------------------------------#
mygpoint = 16
ds['rsd'][0, mygpoint,:,indy, indx][-1] # ground value
ds['rsdcsaf'][0, mygpoint,:,indy, indx][-1] # ground value
ds['rsdcsaf_direct'][0, mygpoint,:,indy, indx][-1] # ground value
diffuse =ds['rsdcsaf'][0, mygpoint,:,indy, indx][-1] - ds['rsdcsaf_direct'][0,mygpoint,:,indy, indx][-1]
ds['rsdcsaf'][0, mygpoint,:,indy, indx][-1] # ground value
ds['rsu'].shape
od = ds['gas_shortwave_optical_depth']

ds['rsdcsaf'][:].shape

plt.figure()
plt.plot(ds['rsdcsaf_direct'][0, :,-1,indy, indx])
plt.plot(ds['rsdcsaf_direct'][0, :,0,indy, indx])
plt.show()


adir = ds['cvisrfgd_dir']
adif = ds['cvisrfgd_dif']

myadir = ds['cvisrfgd_dir'][0, indy, indx]
myadif = ds['cvisrfgd_dif'][0, indy, indx]
mycosz = ds['cosz'][0, indy, indx]
print(myadif, myadir, mycosz)

cosz = ds['cosz']
cosz.shape

pressm = ds['pressm'][:]

pressm.shape

pp = pressm[0,:,10, 30]

yt = ds['grid_yt']
xt = ds['grid_xt']

coszp = cosz[0,:,:].copy()
coszp[coszp<0.001] = np.nan

plt.figure()
plt.pcolormesh(xt, yt, coszp)
plt.colorbar()
plt.show()

plt.figure()
plt.pcolormesh(xt, yt, adif[0,:,:])
plt.colorbar()
plt.show()

plt.figure()
plt.pcolormesh(xt, yt, adir[0,:,:])
plt.colorbar()
plt.show()

plt.figure()
plt.plot([0,1], [0,1], 'k')
plt.plot(np.ravel(adir), np.ravel(adif), 'o')
plt.xlabel('Direct albedo')
plt.ylabel('Diffuse albedo')
plt.show()


for keyi in list(ds.variables):
    if keyi[:2] == 'rs':
        print('__{}__'.format(keyi))
        print(ds[keyi])

# np.sum(dz[0,:,0,0])
dz = ds['dz'][:]
dz.shape
nlevels = 34
nlats = 90
nlons = 144
z = np.zeros((1, nlevels, nlats, nlons))
for i in range(2,nlevels+1):
    print(i)
    z[0,nlevels - i,:,:] = z[0,nlevels-i+1,:,:] + dz[0,nlevels-i,:,:]


np.shape(z[0,:,10,10]    )



# sum over 32 gpoints
rsd_dir = np.sum(ds['rsd_direct'][:], axis=1)
rsd_tot = np.sum(ds['rsd'][:], axis=1)
rsd_dif = rsd_tot - rsd_dir

plt.figure()
plt.plot(rsd_dir[0, :, 10, 10], z[0, :, 10, 10])
plt.plot(rsd_dif[0, :, 10, 10], z[0, :, 10, 10])
plt.xlabel('z [m]')
plt.xlabel('Flux [W/m2]')
plt.show()

# FIRST VALIDATION EXPERIMENT
# do single gpoint (1st of 1st band)
# clear sky aerosol free (csaf)

rsdcsaf = ds['rsdcsaf'][0,:,:]
rsucsaf = ds['rsucsaf']
rsdcsaf.shape
# np.max(z[0,:,10,10])
# np.sum(dz[0,:,10,10])



res = atm.init_atmosphere_gfdl(metadata, mygpoint=0)

# res.keys()
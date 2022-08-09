#

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geospatialtools.gdal_tools as gdal_tools
from osgeo import gdal
import demfuncs as dem
import xarray as xr
from topocalc.viewf import viewf
from topocalc.gradient import gradient_d8

import test_gdal_write_rster as ras

# read the data from UCLA and make sure it is consistent with that I compute and with that from NATE

# location for which I want to compare values
mylon = 12.5; mylat = 46.5
# mylon = 360 - 72.5; mylat = -13.5

# read ucla dataset
# parentdir = '/lustre/f2/dev/Enrico.Zorzetto/landmod/'
parentdir = '/Users/ez6263/Google Drive/My Drive/Ray_rad_data/UCLA_3D_Topo_Data/'
os.listdir(parentdir)
ucladatadir1 = os.path.join(parentdir,'topo_3d_0.23x0.31_c150322.nc')
ucladatadir2 = os.path.join(parentdir,'topo_3d_0.47x0.63_c150322.nc')
ucladatadir3 = os.path.join(parentdir,'topo_3d_0.9x1.25_c150322.nc')
ucladatadir4 = os.path.join(parentdir,'topo_3d_1.9x2.5_c150322.nc')
dsu1 = xr.open_dataset(ucladatadir1)
dsu2 = xr.open_dataset(ucladatadir2)
dsu3 = xr.open_dataset(ucladatadir3)
dsu4 = xr.open_dataset(ucladatadir4)
dsu1EA = dsu1.where( np.logical_and( np.abs(dsu1.LATIXY - mylat) < 0.6,
                                np.abs(dsu1.LONGXY - mylon) < 0.6 ), drop=True)
dsu2EA = dsu2.where( np.logical_and( np.abs(dsu2.LATIXY - mylat) < 0.6,
                                np.abs(dsu2.LONGXY - mylon) < 0.6 ), drop=True)
dsu3EA = dsu3.where( np.logical_and( np.abs(dsu3.LATIXY - mylat) < 0.6,
                                np.abs(dsu3.LONGXY - mylon) < 0.6 ), drop=True)
dsu4EA = dsu4.where( np.logical_and( np.abs(dsu4.LATIXY - mylat) < 1.0,
                                np.abs(dsu4.LONGXY - mylon) < 1.0 ), drop=True)

# np.min(dsu1.LONGXY.values)
# np.max(dsu1.LONGXY.values)
#
# plt.figure()
# plt.pcolormesh(dsu4['LONGXY'], dsu4['LATIXY'], dsu4['SKY_VIEW'])
# plt.show()

dsu4EA['LATIXY'].values
dsu4EA['LONGXY'].values

dsu3EA['LATIXY'].values
dsu3EA['LONGXY'].values


UCLA1_SVF=np.mean(dsu1EA['SKY_VIEW'].values)
UCLA1_TCF=np.mean(dsu1EA['TERRAIN_CONFIG'].values)
UCLA1_SC =np.mean(dsu1EA['SINSL_COSAS'].values)
UCLA1_SS =np.mean(dsu1EA['SINSL_SINAS'].values)

UCLA2_SVF=np.mean(dsu2EA['SKY_VIEW'].values)
UCLA2_TCF=np.mean(dsu2EA['TERRAIN_CONFIG'].values)
UCLA2_SC =np.mean(dsu2EA['SINSL_COSAS'].values)
UCLA2_SS =np.mean(dsu2EA['SINSL_SINAS'].values)

UCLA3_SVF =np.mean(dsu3EA['SKY_VIEW'].values)
UCLA3_TCF =np.mean(dsu3EA['TERRAIN_CONFIG'].values)
UCLA3_SC =np.mean(dsu3EA['SINSL_COSAS'].values)
UCLA3_SS =np.mean(dsu3EA['SINSL_SINAS'].values)

UCLA4_SVF=np.mean(dsu4EA['SKY_VIEW'].values)
UCLA4_TCF=np.mean(dsu4EA['TERRAIN_CONFIG'].values)
UCLA4_SC =np.mean(dsu4EA['SINSL_COSAS'].values)
UCLA4_SS =np.mean(dsu4EA['SINSL_SINAS'].values)

UCLA_SVF = np.array( [UCLA1_SVF, UCLA2_SVF, UCLA3_SVF, UCLA4_SVF]  )
UCLA_TCF = np.array( [UCLA1_TCF, UCLA2_TCF, UCLA3_TCF, UCLA4_TCF]  )
UCLA_SC =  np.array( [UCLA1_SC, UCLA2_SC, UCLA3_SC, UCLA4_SC]  )
UCLA_SS =  np.array( [UCLA1_SS, UCLA2_SS, UCLA3_SS, UCLA4_SS]  )

print(dsu1EA['SKY_VIEW'].shape)
print(dsu2EA['SKY_VIEW'].shape)
print(dsu3EA['SKY_VIEW'].shape)
print(dsu4EA['SKY_VIEW'].shape)
print(np.mean(dsu1EA['SKY_VIEW'].values))
print(np.mean(dsu2EA['SKY_VIEW'].values))
print(np.mean(dsu3EA['SKY_VIEW'].values))
print(np.mean(dsu4EA['SKY_VIEW'].values))

print(dsu1EA['TERRAIN_CONFIG'].shape)
print(dsu2EA['TERRAIN_CONFIG'].shape)
print(dsu3EA['TERRAIN_CONFIG'].shape)
print(dsu4EA['TERRAIN_CONFIG'].shape)
print(np.mean(dsu1EA['TERRAIN_CONFIG'].values))
print(np.mean(dsu2EA['TERRAIN_CONFIG'].values))
print(np.mean(dsu3EA['TERRAIN_CONFIG'].values))
print(np.mean(dsu4EA['TERRAIN_CONFIG'].values))

print('SINSL_COSAS = {:.4f}'.format(np.mean(dsu1EA['SINSL_COSAS'].values)))
print('SINSL_COSAS = {:.4f}'.format(np.mean(dsu2EA['SINSL_COSAS'].values)))
print('SINSL_COSAS = {:.4f}'.format(np.mean(dsu3EA['SINSL_COSAS'].values)))
print('SINSL_COSAS = {:.4f}'.format(np.mean(dsu4EA['SINSL_COSAS'].values)))

print('SINSL_SINAS = {:.4f}'.format(np.mean(dsu1EA['SINSL_SINAS'].values)))
print('SINSL_SINAS = {:.4f}'.format(np.mean(dsu2EA['SINSL_SINAS'].values)))
print('SINSL_SINAS = {:.4f}'.format(np.mean(dsu3EA['SINSL_SINAS'].values)))
print('SINSL_SINAS = {:.4f}'.format(np.mean(dsu4EA['SINSL_SINAS'].values)))


# EQUAL AREA MOLLWEIDE PROJECTION
eaproj = '+proj=moll +lon_0=%.16f +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs'




# FIRST, READ RMC RESULTS IN LAT LON
print( os.system("which gdalwarp"))

simul_folder = os.path.join('/Users', 'ez6263', 'Documents','dem_datasets')   # PRINCETON LAPTOP
simul_name = 'output_cluster_PP3D_EastAlps_run1_a17'
# simul_name = 'output_cluster_PP3D_Peru_run1_a35'

outdir = os.path.join(simul_folder, 'output_maps')
if not os.path.exists(outdir): os.makedirs(outdir)

os.listdir(simul_folder)
simdir = os.path.join(simul_folder, simul_name, 'output_sim', "output_sim_3D")
dfc = pd.read_csv(os.path.join(simul_folder, simul_name, 'list_sim_cases_3D.csv'))
cases = dfc['cases'].values
ds_filename = os.path.join(simdir, 'photonmc_output_{}.nc'.format(cases[0]))
ds = xr.open_dataset(ds_filename)
# READ AND SAVE TERRAIN VARIABLES:
y0 = ds['lat'].values
x0 = ds['lon'].values
Z0 = ds['elev'].values
dem_latlon0 = ds['elev'].values # 1211*1211
# minlat = 46.0
# maxlat = 47.0
# minlon = 12.0
# maxlon = 13.0

dem_latlon = np.rot90(dem_latlon0)

# LONL = np.array([12.0, 13.0])
# LATL = np.array([46.0, 47.0])


# PERU
LONL = np.array([  min(mylon + 0.5, mylon - 0.5), max(mylon + 0.5, mylon - 0.5)])
LATL = np.array([  min(mylat + 0.5, mylat - 0.5), max(mylat + 0.5, mylat - 0.5)])


# write to raster
# gdal_tools.write_raster(outdem1, )
outfile1 = '%s/demll.tif' % outdir
ras.write_raster_WGS84(outfile1, dem_latlon, LATL, LONL, nodata = -9999)

demll = gdal_tools.read_data(outfile1).data
# interp can be one of: bilinear, cubic, cubicspline, average, lanczos ...
infile_ll = outfile1
outfile_ea = '%s/demea.tif' % outdir
logfile_ea = '%s/logea.log' % outdir
eares = 90.0

ras.write_ea(infile_ll, outfile_ea, logfile_ea, eares=eares, interp='lanczos')
demea_raster = gdal_tools.read_data(outfile_ea)
demea = demea_raster.data
print('ea shape:: ny = {}, nx = {}'.format(demea_raster.ny, demea_raster.nx))
# check the dem for artifacts
plt.figure()
plt.imshow(demea)
plt.show()

# ras.write_ea(infile_ll, outfile_ea, logfile_ea, eares=eares, interp='average')

# try different interpolation methods::
# INTERP = ['near', 'average', 'cubic', 'lanczos', 'bilinear', 'cubicspline', 'rms']
INTERP = ['near', 'average', 'cubic', 'lanczos', 'bilinear', 'cubicspline']
# INTERP = ['near', 'average', 'cubic', 'lanczos']
ninterp = len(INTERP)

SCAEA = np.zeros(ninterp)
SSAEA = np.zeros(ninterp)
SVFEA= np.zeros(ninterp)
TCFEA= np.zeros(ninterp)

SCAEA_CAP = np.zeros(ninterp)
SSAEA_CAP = np.zeros(ninterp)
SVFEA_CAP= np.zeros(ninterp)
TCFEA_CAP= np.zeros(ninterp)

# COMPUTE SVF / TCF FROM LL FILE:::
nangles = 32
dx = 60.0
dy = 90.0
dxy = np.sqrt(dx * dy)
svf0ll, tcf0ll = viewf(demll.astype(np.float64), spacing=dxy, nangles=nangles)
# instead of rot, switch dx and dy  - faster here
slope0ll, aspect0ll = gradient_d8(demll, dx, dy, aspect_rad=True)
cossll = np.cos(slope0ll)
sinsll = np.sin(slope0ll)
svfll = svf0ll / cossll
tcfll = tcf0ll / cossll
cossll_cap = cossll.copy()
sinsll_cap = sinsll.copy()

buf = 100
fix_cosval = 0.4
angle = np.arccos(fix_cosval)*180.0/np.pi
fix_sinval = np.sin( np.arccos(fix_cosval))*100

cossll_cap[cossll_cap < fix_cosval] = fix_cosval
sinsll_cap[sinsll_cap > fix_sinval] = fix_sinval

svfll_cap = svf0ll / cossll_cap
tcfll_cap = tcf0ll / cossll_cap

# sinscosall = np.tan(slope0ll)*np.cos(aspect0ll)
# sinssinall = np.tan(slope0ll)*np.sin(aspect0ll)

sinscosall = sinsll/cossll*np.cos(aspect0ll)
sinssinall = sinsll/cossll*np.sin(aspect0ll)

sinscosall_cap = sinsll_cap/cossll_cap*np.cos(aspect0ll)
sinssinall_cap = sinsll_cap/cossll_cap*np.sin(aspect0ll)

svfll = svfll[buf:-buf, buf:-buf]
tcfll = tcfll[buf:-buf, buf:-buf]
sinscosall = sinscosall[buf:-buf, buf:-buf]
sinssinall = sinssinall[buf:-buf, buf:-buf]

svfll_cap = svfll_cap[buf:-buf, buf:-buf]
tcfll_cap = tcfll_cap[buf:-buf, buf:-buf]
sinscosall_cap = sinscosall_cap[buf:-buf, buf:-buf]
sinssinall_cap = sinssinall_cap[buf:-buf, buf:-buf]

SCALL = np.ones(ninterp)*np.mean( sinscosall )
SSALL = np.ones(ninterp)*np.mean( sinssinall )
SVFLL=  np.ones(ninterp)*np.mean( svfll )
TCFLL=  np.ones(ninterp)*np.mean( tcfll )

SVFLL_CAP= np.ones(ninterp)* np.mean( svfll_cap )
TCFLL_CAP= np.ones(ninterp)* np.mean( tcfll_cap )
SCALL_CAP=np.ones(ninterp)* np.mean( sinscosall_cap )
SSALL_CAP=np.ones(ninterp)* np.mean( sinssinall_cap )


for ii, interp in enumerate(INTERP):
    print('interp = {}'.format(interp))
    outfile_ea_ii = '%s/demea_%s.tif' % (outdir, interp)
    ras.write_ea(infile_ll, outfile_ea_ii, logfile_ea, eares=eares, interp=interp)
    demea_raster = gdal_tools.read_data(outfile_ea_ii)
    demea = demea_raster.data
    print('ea shape:: ny = {}, nx = {}'.format(demea_raster.ny, demea_raster.nx))
    # write_ea(infile_ll, outfile_ea, logfile_ea, eares=eares)




    svf0ea, tcf0ea = viewf(demea.astype(np.float64), spacing=eares, nangles=nangles)
    slope0ea, aspect0ea = gradient_d8(demea, eares, eares, aspect_rad=True)
    # cossea = np.cos(slope0ea)

    cossea = np.cos(slope0ea)
    sinsea = np.sin(slope0ea)
    # cap the slope to avoid artifacts
    cossea_cap = cossea.copy()
    sinsea_cap = sinsea.copy()

    cossea_cap[cossea_cap < fix_cosval] = fix_cosval
    sinsea_cap[sinsea_cap > fix_sinval] = fix_sinval
    # cossea_cap[cossea_cap < 0.2] = 0.2
    # print(np.arccos(0.2)*180.0/np.pi)
    # print(np.tan( np.arccos(0.2))*100)
    svfea = svf0ea/cossea
    tcfea = tcf0ea/cossea
    # cap the slope angles at ~ 80 degrees
    svfea_cap = svf0ea/cossea_cap
    tcfea_cap = tcf0ea/cossea_cap

    sinscosaea = np.tan(slope0ea) * np.cos(aspect0ea)
    sinssinaea = np.tan(slope0ea) * np.sin(aspect0ea)
    sinscosaea_cap = sinsea_cap / cossea_cap * np.cos(aspect0ea)
    sinssinaea_cap = sinsea_cap / cossea_cap * np.sin(aspect0ea)

    # remove buffers at the boudnaries
    # print(svfea.shape)
    svfea = svfea[buf:-buf, buf:-buf]
    svf0ea = svf0ea[buf:-buf, buf:-buf]
    svfll = svfll[buf:-buf, buf:-buf]
    svf0ll = svf0ll[buf:-buf, buf:-buf]

    svfea_cap = svfea_cap[buf:-buf, buf:-buf]
    svfll_cap = svfll_cap[buf:-buf, buf:-buf]

    sinscosaea = sinscosaea[buf:-buf, buf:-buf]
    sinssinaea = sinssinaea[buf:-buf, buf:-buf]
    sinscosaea_cap = sinscosaea_cap[buf:-buf, buf:-buf]
    sinssinaea_cap = sinssinaea_cap[buf:-buf, buf:-buf]



    SCAEA[ii] = np.mean( sinscosaea )
    SSAEA[ii] = np.mean( sinssinaea )
    SVFEA[ii] = np.mean( svfea )
    TCFEA[ii] = np.mean( tcfea )

    SCAEA_CAP[ii] = np.mean(sinscosaea_cap)
    SSAEA_CAP[ii] = np.mean(sinssinaea_cap)
    SVFEA_CAP[ii] = np.mean(svfea_cap)
    TCFEA_CAP[ii] = np.mean(tcfea_cap)



fig, ax = plt.subplots(2,2)

ax[0,0].set_title('svf')
ax[0,0].plot(SVFEA, '-or')
# ax[0,0].plot(SVFEA_CAP, '--og')
ax[0,0].plot(SVFLL, '-k')
# ax[0,0].plot(SVFLL_CAP, '--b')
ax[0,0].plot( UCLA_SVF, '-.c')
# ax[0,0].plot( np.ones(ninterp)*UCLA1_SVF, '-.c')
# ax[0,0].plot( np.ones(ninterp)*UCLA2_SVF, '--c')
# ax[0,0].plot( np.ones(ninterp)*UCLA3_SVF, '-.g')
# ax[0,0].plot( np.ones(ninterp)*UCLA4_SVF, '--g')
ax[0,0].set_xticks(np.arange(len(SVFEA)))
ax[0,0].set_xticklabels(INTERP, rotation='vertical', fontsize=12)

ax[0,1].set_title('tcf')
ax[0,1].plot(TCFEA, '-or')
# ax[0,1].plot(TCFEA_CAP, '--og')
ax[0,1].plot(TCFLL, '-k')
# ax[0,1].plot(TCFLL_CAP, '--b')
ax[0,1].plot( UCLA_TCF, '-.c')
# ax[0,0].plot( np.ones(ninterp)*UCLA1_TCF, '-.c')
# ax[0,0].plot( np.ones(ninterp)*UCLA2_TCF, '--c')
# ax[0,1].plot( np.ones(ninterp)*UCLA3_TCF, '-.g')
# ax[0,1].plot( np.ones(ninterp)*UCLA4_TCF, '--g')
ax[0,1].set_xticks(np.arange(len(TCFEA)))
ax[0,1].set_xticklabels(INTERP, rotation='vertical', fontsize=12)
ax[1,0].set_title('sinscosa')
ax[1,0].plot(SCAEA, '-or')
# ax[1,0].plot(SCAEA_CAP, '--og')
ax[1,0].plot(SCALL, '-k')
# ax[1,0].plot(SCALL_CAP, '--b')

ax[1,0].plot( -UCLA_SS, '-.c')
# ax[1,0].plot( -np.ones(ninterp)*UCLA1_SS, '-.c')
# ax[1,0].plot( -np.ones(ninterp)*UCLA2_SS, '--c')
# ax[1,0].plot( -np.ones(ninterp)*UCLA3_SS, '-.g')
# ax[1,0].plot( -np.ones(ninterp)*UCLA4_SS, '--g')
ax[1,0].set_xticks(np.arange(len(SCAEA)))
ax[1,0].set_xticklabels(INTERP, rotation='vertical', fontsize=12)
ax[1,1].set_title('sinssina')
ax[1,1].plot(SSAEA, '-or')
# ax[1,1].plot(SSAEA_CAP, '--og')
ax[1,1].plot(SSALL, '-k')
# ax[1,1].plot(SSALL_CAP, '--b')
ax[1,1].plot( UCLA_SC, '-.c')
# ax[1,1].plot( np.ones(ninterp)*UCLA1_SC, '-.c')
# ax[1,1].plot( np.ones(ninterp)*UCLA2_SC, '--c')
# ax[1,1].plot( np.ones(ninterp)*UCLA3_SC, '-.g')
# ax[1,1].plot( np.ones(ninterp)*UCLA4_SC, '--g')
ax[1,1].set_xticks(np.arange(len(SSAEA)))
ax[1,1].set_xticklabels(INTERP, rotation='vertical', fontsize=12)

plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(sinssinall, vmax=1, vmin=-1)
# plt.imshow(sinscosall, vmax=1, vmin=-1)
plt.show()


# np.std(sinscosall)
# np.std(sinscosaea)
#
# (UCLA_SC - SCALL)/np.std(sinscosaea)
# (UCLA_SS - SSALL)/np.std(sinssinaea)



# dv = {"svfea":svfea, "svfll":svfll, "svf0ea":svf0ea, "svf0ll":svf0ll, "cossea":cossea, "cossll":cossll}
dv = {"svfea":svfea, "svfll":svfll, "svf0ea":svf0ea, "svf0ll":svf0ll, "cossea":cossea, "cossll":cossll,
      "svfea_cap":svfea_cap, "svfll_cap":svfll_cap}
# dprint(dv)

# np.mean(svfea) - np.mean(svfll)
# # np.mean(svfea[svfea < 2]) - np.mean(svfll)
# (np.mean(svfea) - np.mean(svfll))/np.std(svfea)
# (np.mean(svfea) - np.mean(svfll))/np.std(svfll)

# #
# plt.figure()
# plt.hist(np.ravel(svfll), bins=60, density = True)
# plt.hist(np.ravel(svfea), bins=600, density = True, alpha = 0.5)
# plt.xlim([0, 3])
# plt.show()
#
#
# # plt.figure()
# # plt.plot(np.ravel(svfll), np.ravel(svfll), '--k')
# # plt.hist(np.ravel(svfll), np.ravel(svfea), 'or')
# # # plt.xlim([0, 3])
# # plt.show()
#
#
# plt.figure()
# plt.hist(np.ravel(cossll), bins=30, density = True)
# plt.hist(np.ravel(cossea), bins=30, density = True, alpha = 0.5)
# plt.xlim([0, 1])
# plt.show()
#
#
# #
# plt.figure()
# plt.hist(np.ravel(svf0ll), density = True, bins=30)
# plt.hist(np.ravel(svf0ea), density = True, bins=30, alpha = 0.5)
# plt.show()
#

# Now, convert raster to equal area projection and re-compute terrain variables
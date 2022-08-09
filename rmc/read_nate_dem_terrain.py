
import os
import numpy as np
import pickle
from geospatialtools import gdal_tools
import matplotlib.pyplot as plt
import photon_mc_land







# datadir = os.path.join('/Users/ez6263/Documents/rmc_datasets/Nates_dems')
datadir = os.path.join('/Users/ez6263/Documents/rmc_datasets/GFDL_preproc_dems')
datadir_temp = os.path.join(datadir, '..', 'preprocessed_dem')
os.system("mkdir -p {}".format(datadir_temp))
# os.listdir(datadir)
# domain = 'Nepal'
# domain = 'Peru'
# domain = 'EastAlps'
domain = 'FrenchAlps'


cbufferx = 0.45; cbuffery = 0.45 # crop original lat-lon file (buffer ion lat-lonm coords)
peoffsetx = 20; peoffsety = 20 # complete periodic edges
eares = 90.0 # target resolution [m] of equal area map

# READ DEM IN LATLON
demfile = os.path.join( datadir, '{}_dem_latlon.tif'.format(domain))
data = gdal_tools.read_data(demfile)
print(data.nx)
print(data.ny)
print(data.projection)
print(data.proj4)
print(data.nodata)

print(data.minx, data.maxx)
print(data.miny, data.maxy)
print('res:', data.resx, data.resy)

# plt.figure()
# plt.imshow(data.data, vmin = 0)
# # #plt.imshow(  np.rot90(np.fliplr(np.flipud(data.data))), vmin = 0)
# # plt.imshow(  np.rot90(data.data, k=-1), vmin = 0)
# plt.show()

# CROP DATASET TO DESIRED LATLON BOUNDING BOX
# minx_crop = 12.2; maxx_crop = 12.8; miny_crop = 46.2; maxy_crop = 46.8
minx_crop = data.minx + cbufferx; maxx_crop = data.maxx - cbufferx
miny_crop = data.miny + cbuffery; maxy_crop = data.maxy - cbuffery

infile_ll_cropped = os.path.join(datadir, '{}_dem_latlon.tif'.format(domain))
outfile_ll_cropped = os.path.join(datadir_temp, '{}_dem_latlon_cropped.tif'.format(domain))
logfile_ll_cropped = os.path.join(datadir_temp, '{}_dem_latlon_cropped.log'.format(domain))

os.system('rm -f {} {}'.format(logfile_ll_cropped, outfile_ll_cropped))
os.system("gdalwarp -te %.16f %.16f %.16f %.16f %s %s >& %s" % (
    minx_crop, miny_crop, maxx_crop, maxy_crop,
    infile_ll_cropped, outfile_ll_cropped, logfile_ll_cropped))

data_ll_cropped = gdal_tools.read_data(outfile_ll_cropped)
# plt.figure()
# plt.imshow(data_ll_cropped.data, vmin = 0)
# plt.show()


# REPROJECT CROPPED DEM TO EQUAL AREA PROJECTION [MOLLEWEIDE]

infile_ea_cropped = os.path.join(datadir_temp, '{}_dem_latlon_cropped.tif'.format(domain))
outfile_ea_cropped = os.path.join(datadir_temp, '{}_dem_ea_cropped.tif'.format(domain))
logfile_ea_cropped = os.path.join(datadir_temp, '{}_dem_ea_cropped.log'.format(domain))
eaproj = '+proj=moll +lon_0=%.16f +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs'

minlon = data_ll_cropped.minx; maxlon = data_ll_cropped.maxx
lproj = eaproj % float((maxlon+minlon)/2)
os.system("rm -f {} {}".format(outfile_ea_cropped, logfile_ea_cropped))
os.system('gdalwarp -r average -dstnodata -9999 -tr %.16f %.16f -t_srs "%s" %s %s >& %s'
          % (eares, eares, lproj, infile_ea_cropped, outfile_ea_cropped, logfile_ea_cropped))

data_ea_cropped = gdal_tools.read_data(outfile_ea_cropped)


# GET COORDINATE AND ARRAY FOR CROPPED EA DEM
nx = data_ea_cropped.nx
ny = data_ea_cropped.ny

# APPROXIMATE - GET LATLON COORDINATES FROM FILE WITH SAME ~ CROP BOUNDS
lons0 = np.linspace(data_ll_cropped.minx, data_ll_cropped.maxx, data_ea_cropped.nx)
lats0 = np.linspace(data_ll_cropped.miny, data_ll_cropped.maxy, data_ea_cropped.ny)
# x0 = np.arange(data_ea_cropped.nx)*eares
# y0 = np.arange(data_ea_cropped.ny)*eares

print(data_ea_cropped.data.shape)
print(lons0.shape, lats0.shape)
# print(x0.shape, y0.shape)

# ROTATE MATRIX ACCORDING TO RMC CONVENTION
Z0 =  np.rot90( data_ea_cropped.data, k=-1)

# print(np.min(data_ea_cropped.data))
# plt.figure()
# # plt.imshow(data_ea_cropped.data, vmin = 0)
# plt.imshow(Z0)
# plt.show()


# FURTHER CROP TO GET RID OF NODATA AT THE BOUNDARIES
nbx, nby = np.shape(Z0)
print('shape of Z0 = {}'.format( np.shape(Z0)))
bfdx = 12 # EA min buffer to avoid missing data (EA:11, PE:6, NEP:7, FREA: 8)
bfdy = 0 # No need to crop in y for all
Z = Z0[bfdx:nbx-bfdx, bfdy:nby-bfdy].copy()
# x = x0[bfdx:nbx-bfdx].copy()
# y = y0[bfdy:nby-bfdy].copy()
lons = lons0[bfdx:nbx-bfdx].copy()
lats = lats0[bfdy:nby-bfdy].copy()

nbx2, nby2 = np.shape(Z)
print('shape of Z = {}'.format( np.shape(Z)))

x = np.arange(nbx2)*eares
y = np.arange(nby2)*eares


# AA = np.arange(16).reshape(4,4)
#


# X, Y = np.meshgrid(x, y)
# LONS, LATS = np.meshgrid(lons, lats)
#
# Y, X = np.meshgrid(y, x)
# LATS, LONS = np.meshgrid(lats, lons)
#
# plt.figure()
# plt.pcolormesh(LONS, LATS, Z)
# # plt.pcolormesh(X, Y, Z)
# plt.show()

# np.min(Z[:,0])
# np.min(Z[:,-1])
# np.min(Z[0,:])
# np.min(Z[-1 ,:])

# print(np.min(Z))

xnb, ynb, lonnb, latnb, ZNB = photon_mc_land.complete_periodic_edges_ea(
    x, y, lons, lats, Z, offsetx=peoffsety, offsety=peoffsety)

#
# plt.figure()
# plt.imshow(np.rot90(Z, k=1))
# plt.show()


YNB, XNB = np.meshgrid(ynb, xnb)
LATNB, LONNB = np.meshgrid(latnb, lonnb)
plt.figure()
# plt.pcolormesh(LONNB, LATNB, ZNB, shading='nearest')
plt.pcolormesh(XNB, YNB, ZNB, shading='auto')
plt.show()

print( ZNB.shape, lonnb.shape, latnb.shape)
print( lonnb[0], lonnb[-1], latnb[0], latnb[-1])


# WRITE RASTER WITH RESULTING DEM (NAH..)

# PICKLE RESULTS TO READ THEM IN main_photon_cluster
resdict = {'y':ynb, 'x':xnb, 'lats':latnb, 'lons':lonnb, 'Z':ZNB,
           'proj':"equal_area", 'eares':eares,
           'peoffsetx':peoffsetx, 'peoffesety':peoffsety
           }

output_file = os.path.join(datadir_temp, '{}_dem_input.pkl'.format(domain))
with open(output_file, 'wb') as picklef:
    pickle.dump(resdict, picklef)

# read back the saved file
with open(output_file, "rb") as input_file:
    resdict2 = pickle.load(input_file)

print(resdict2.keys())
print(np.mean(resdict2['Z']))
print(np.mean(resdict['Z']))

print(resdict2['x'][0])
print(resdict2['y'][1])
print(resdict2['Z'][1])


print(resdict['x'][0])
print(resdict['y'][1])
print(resdict['Z'][1])

import pickle
import os
import netCDF4 as nc
import osgeo.ogr as ogr
import numpy as np
import json
#metadata = pickle.load(open('metadata.pck'))
import sys
import geospatialtools.gdal_tools as gdal_tools
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
mdfile = sys.argv[1]
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))
rdir = '%s/river' % metadata['dir']
file = '%s/hydrography.tile1.nc' % rdir
file_update = '%s/river_data_update.nc' % rdir #Moved to river_data_update
#file_update = file
os.system('cp %s %s' % (file,file_update))
fp = nc.Dataset(file_update,'a')
#fp = nc.Dataset(file,'a')
olfrac = fp.variables['land_frac'][:]
grid_x = fp.variables['grid_x'][:]
grid_y = fp.variables['grid_y'][:]

#lats = fp.variables[
#Open access to the shapefile
file_shp = '%s/shapefile/grid.shp' % metadata['dir']
driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open(file_shp, 0)

#Iterate through all grid cells and create their database
layer = ds.GetLayer()
#ils = np.arange(len(layer))
for il in range(len(layer)):
 feature = layer[il]
 id = feature.GetField("ID")
 y = feature.GetField("Y")
 x = feature.GetField("X")
 tile = feature.GetField("TILE")
 lfrac = feature.GetField("LFN")
 bbox = feature.GetGeometryRef().GetEnvelope()
 print('id:%d,tile:%d,x:%d,y:%d' % (id,tile,x,y))
 #if olfrac[x-1,y-1]==0:
 print(lfrac,olfrac[y-1,x-1])#,grid_x[x-1],grid_y[y-1],lon,lat
 # lfrac = 0.0 #TEMPORARY
 #if olfrac[y-1,x-1] == 0.0:continue
 olfrac[y-1,x-1] = lfrac
 #lat = (bbox[2]+bbox[3])/2
 #lon = (bbox[0]+bbox[1])/2+360

#Regrid the meteo mask
'''
minlon = metadata['minlon'] - 360
maxlon = metadata['maxlon'] - 360
minlat = metadata['minlat']
maxlat = metadata['maxlat']
file = 'workspace/mmask.tif'
os.system('rm %s' % file)
os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f %s %s' % (minlon,minlat,maxlon,maxlat,metadata['meteomask'],file))
mmask = np.flipud(gdal_tools.read_raster(file))
mmask[mmask > 0] = 1
mmask[mmask < 0] = 0
omask = np.copy(olfrac)
omask[omask > 0] = 1
omask[omask == 0] = 0
diff = omask - mmask
print np.where(diff == 1)
olfrac[diff == 1] = 0.0
print mmask[22,162]#,22]
print omask[22,162]#,22]
print olfrac[22,162]#,22]
#olfrac[22,162] = 0.0 #HACK!!!!!!!'''
#Read in the meteo mask for this grid and use to correct
fp.variables['land_frac'][:] = olfrac[:]
fp.close()

#If necessary set file to be the river data
if metadata['land_fractions'] != 'original':
 print("Replacing the original river data with the updated one")
 os.system('mv %s %s' % (file_update,file))

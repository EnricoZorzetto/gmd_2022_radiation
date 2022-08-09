import h5py
import geospatialtools.gdal_tools as gdal_tools
import numpy as np
#import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import time
import sys
import pickle
mdfile = sys.argv[1]

 return

#Define the parameters
undef = -9999.0
iyear = 2002
fyear = 2002

#Read in the metadata
metadata = pickle.load(open(mdfile))
dir = metadata['dir']
size = metadata['npes']

#Create meteorology directory
mdir = '%s/meteorology' % dir
os.system('mkdir -p %s' % mdir)
os.system('mkdir -p %s/workspace' % mdir)

#Create the meteorology mapping
print("Creating the meteorology mapping")
create_meteorology_mapping(mdir)

#Create the data for the grid cell
print("Extracting tile meteorology for each cell")
extract_data

#Open access to the land database
file = '%s/land/land_model_input_database.h5' % dir
fp = h5py.File(file)
for cell in fp['grid_data']:
 print cell
 #Retrieve the metadata for the domain
 file = '%s/%s/tiles.tif' % (dir,cell)
 md = gdal_tools.retrieve_metadata(file)
 minlat = md['miny']
 minlon = md['minx']
 maxlat = md['maxy']
 maxlon = md['maxx']
 res = abs(md['resx'])
 #Map the is/js to the domain
 ilats_file = '%s/%s/ilats.tif' % (dir,cell)
 ilons_file = '%s/%s/ilons.tif' % (dir,cell)
 cids_file = '%s/%s/cids.tif' % (dir,cell)
 os.system('gdalwarp -overwrite -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s' % (minlon,minlat,maxlon,maxlat,res,res,'ilats.tif',ilats_file))
 os.system('gdalwarp -overwrite -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s' % (minlon,minlat,maxlon,maxlat,res,res,'ilons.tif',ilons_file))
 os.system('gdalwarp -overwrite -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s' % (minlon,minlat,maxlon,maxlat,res,res,'cids.tif',cids_file))
 #Read in the tiles and is/js
 tiles = gdal_tools.read_raster(file)
 ilats = gdal_tools.read_raster(ilats_file)
 ilons = gdal_tools.read_raster(ilons_file)
 cids = gdal_tools.read_raster(cids_file)
 #Iterate through the tiles
 tids = np.unique(tiles)
 tids = tids[tids != undef]
 db = {}
 mdb = {'cids':[],'ilats':[],'ilons':[]}
 for tid in tids:
  m = tiles == tid
  (tcids,idx,counts) = np.unique(cids[m],return_counts=True,return_index=True)
  frac = counts/float(np.sum(counts))
  t_ilats = ilats[m][idx].astype(np.int)
  t_ilons = ilons[m][idx].astype(np.int)
  db[tid] = {'frac':frac,'ilats':t_ilats,'ilons':t_ilons,'cids':cids[m][idx].astype(np.int)}
  for i in xrange(cids[m][idx].size):
   cid = cids[m][idx][i]
   if cid not in mdb['cids']:
    mdb['cids'].append(int(cid))
    mdb['ilats'].append(t_ilats[i])
    mdb['ilons'].append(t_ilons[i])
 #Determine the boundaries to read the data
 minilat = np.min(mdb['ilats'])
 maxilat = np.max(mdb['ilats'])
 minilon = np.min(mdb['ilons'])
 maxilon = np.max(mdb['ilons'])
 #Iterate per month and construct the database
 output = {}
 nt = 0
 for year in xrange(2002,2003):
  for month in xrange(1,3):
   file = '/lustre/f1/unswept/Nathaniel.Chaney/data/PCF/meteorology_conus/200207.nc.gcp'
   #file = '/lustre/f1/unswept/Nathaniel.Chaney/data/PCF/meteorology_conus/test.nc'
   fp = nc.Dataset(file)
   for var in ['lwdown','precip','psurf','spfh','swdown','tair','wind']:
    if var not in output:output[var] = []
    #Extract the data for all the cells in the domain
    data = np.ma.getdata(fp[var][:,minilat:maxilat+1,minilon:maxilon+1])
    #Compute the data for each tile
    for tid in db:
     tmp = np.sum(db[tid]['frac']*data[:,db[tid]['ilats']-minilat,db[tid]['ilons']-minilon],axis=1)
     output[var].append(tmp)
   nt += fp['lwdown'].shape[0]
   fp.close()
 for var in output:
  output[var] = np.array(output[var]).reshape(nt,len(db.keys()))
  #Write the data to the met file

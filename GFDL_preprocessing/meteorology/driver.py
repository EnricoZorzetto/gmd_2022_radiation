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

def create_meteorology_mapping(mdir):

 #Create a mapping of the meteorology grid
 file = '/lustre/f1/unswept/Nathaniel.Chaney/data/PCF/meteorology_conus/PCF.tif'
 #Read in and place is/js
 md = gdal_tools.retrieve_metadata(file)
 data = gdal_tools.read_raster(file)
 ilats = np.arange(data.shape[0])
 ilons = np.arange(data.shape[1])
 ilons_all,ilats_all = np.meshgrid(ilons,ilats)
 cids = np.arange(ilats_all.size).reshape(ilats_all.shape)
 ilats_all[data == undef] = undef
 ilons_all[data == undef] = undef
 cids[data == undef] = undef
 #Write out the is and js
 md['nodata'] = undef
 gdal_tools.write_raster('%s/workspace/ilats.tif' % mdir,md,np.flipud(ilats_all))
 gdal_tools.write_raster('%s/workspace/ilons.tif' % mdir,md,ilons_all)
 gdal_tools.write_raster('%s/workspace/cids.tif' % mdir,md,cids)

 return

def extract_tile_meteorology(mdir,ldir,dir,iyear,fyear,cell):

  #Retrieve the metadata for the domain
  file = '%s/%s/tiles.tif' % (ldir,cell)
  md = gdal_tools.retrieve_metadata(file)
  minlat = md['miny']
  minlon = md['minx']
  maxlat = md['maxy']
  maxlon = md['maxx']
  res = abs(md['resx'])
  #Map the is/js to the domain
  ilats_file = '%s/%s/ilats.tif' % (mdir,cell)
  ilons_file = '%s/%s/ilons.tif' % (mdir,cell)
  cids_file = '%s/%s/cids.tif' % (mdir,cell)
  os.system('gdalwarp -overwrite -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s' % (minlon,minlat,maxlon,maxlat,res,res,'%s/workspace/ilats.tif' % mdir,ilats_file))
  os.system('gdalwarp -overwrite -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s' % (minlon,minlat,maxlon,maxlat,res,res,'%s/workspace/ilons.tif' % mdir,ilons_file))
  os.system('gdalwarp -overwrite -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s' % (minlon,minlat,maxlon,maxlat,res,res,'%s/workspace/cids.tif' % mdir,cids_file))
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
  for year in xrange(iyear,fyear+1): #HERE
   output[year] = {}
   #nt = 0
   for month in xrange(1,13):
    print year,month
    file = '/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/120arcsec/rechunk/%04d%02d.nc' % (year,month) #HERE
    fp = nc.Dataset(file)
    for var in ['lwdown','precip','psurf','spfh','swdown','tair','wind']:
     if var not in output[year]:output[year][var] = []
     #Extract the data for all the cells in the domain
     data = np.ma.getdata(fp[var][:,minilat:maxilat+1,minilon:maxilon+1])
     data[data == 30784.0] = np.mean(data[data != 30784.0])
     #Compute the data for each tile
     tmpall = []
     for tid in tids:
      tmp = np.sum(db[tid]['frac']*data[:,db[tid]['ilats']-minilat,db[tid]['ilons']-minilon],axis=1)
      tmpall.append(tmp)
     tmpall = np.array(tmpall).T
     output[year][var].append(tmpall)
     #print np.shape(tmp)
    #Get time
    if 'time' not in output[year]:
     output[year]['time'] = list(fp['time'][:])
    else:
     output[year]['time'] = output[year]['time'] + list(fp['time'][:] + output[year]['time'][-1] + 60) 
    #nt += fp['lwdown'].shape[0]
    fp.close()
    #tmp = np.copy(tiles).astype(np.float32)
    #tmpall = np.array(output[2002]['precip'])[0,:,:]
    #for tid in tids:
    # tmp[tiles == tid] = np.mean(tmpall[:,tid])
    # #print tid,3*3600*24*365*np.mean(tmpall[:,tid])
    # #print tid,np.mean(tmpall[:,tid])
    #tmp = 3*3600*24*365*np.ma.masked_array(tmp,tmp==-9999)
    #plt.imshow(tmp,interpolation='nearest')
    #plt.colorbar()
    #plt.show()
   for var in output[year]:
    if var not in ['time',]:
     output[year][var] = np.concatenate(output[year][var],axis=0)
    else:
     output[year][var] = np.array(output[year][var])

  return output

def write_tile_meteorology(output,mdir,cell,lat,lon):

 for year in output:

  print year
  #Create the file
  ofile = '%s/%s/%04d.nc' % (mdir,cell,year)
  fpo = nc.Dataset(ofile,'w')

  #Create the dimensions
  fpo.createDimension('time',None)
  #fpo.createDimension('z',1)
  fpo.createDimension('latitude',1)
  fpo.createDimension('longitude',1)
  fpo.createDimension('ptid',output[year]['lwdown'].shape[1])
  
  #Create the variables

  #z
  #fpo.createVariable('z','f8',('z',))
  #fpo['z'].units = "level"
  #fpo['z'][:] = 0.0

  #ptid
  fpo.createVariable('ptid','i4',('ptid',))
  fpo['ptid'].units = "N/A"
  fpo['ptid'][:] = np.arange(output[year]['lwdown'].shape[1])+1

  #latitude
  fpo.createVariable('latitude','f8',('latitude',))
  fpo['latitude'].standard_name = "latitude"
  fpo['latitude'].long_name = "latitude"
  fpo['latitude'].units = "degrees_north"
  fpo['latitude'].axis = "Y"
  fpo['latitude'][:] = lat

  #longitude
  fpo.createVariable('longitude','f8',('longitude',))
  fpo['longitude'].standard_name = "longitude" 
  fpo['longitude'].long_name = "longitude" 
  fpo['longitude'].units = "degrees_east" 
  fpo['longitude'].axis = "X" 
  fpo['longitude'][:] = lon

  #time
  fpo.createVariable('time','f8',('time',))
  fpo['time'].standard_name = "time"
  fpo['time'].long_name = "Time"
  fpo['time'].units = "minutes since %04d-1-1 00:00:00" % year
  fpo['time'].calendar = "gregorian"
  fpo['time'].axis = "T"
  fpo['time'][:] = output[year]['time'][:]

  #dlwrf
  #fpo.createVariable('dlwrf','f4',('time','z','latitude','longitude','ptid'))
  fpo.createVariable('dlwrf','f4',('time','ptid','latitude','longitude'))
  fpo['dlwrf'].long_name = "Downward Longwave Radiation"
  fpo['dlwrf'].missing_value = 30784.0
  fpo['dlwrf'].standard_name = "surface_downwelling_longwave_flux_in_air"
  fpo['dlwrf'].units = "W/m^2"
  fpo['dlwrf'][:,:,0,0]  = output[year]['lwdown']

  #dswrf
  fpo.createVariable('dswrf','f4',('time','ptid','latitude','longitude'))
  fpo['dswrf'].long_name = "Downward Shortwave Radiation"
  fpo['dswrf'].missing_value = 30784.0
  fpo['dswrf'].standard_name = "surface_downwelling_shortwave_flux_in_air"
  fpo['dswrf'].units = "W/m^2"
  fpo['dswrf'][:,:,0,0]  = output[year]['swdown']

  #prcp
  fpo.createVariable('prcp','f4',('time','ptid','latitude','longitude'))
  fpo['prcp'].long_name = "Precipitation"
  fpo['prcp'].missing_value = 30784.0
  fpo['prcp'].standard_name = "precipitation_flux"
  fpo['prcp'].units = "kg/m^2/s"
  #Correct the precip
  tmp = output[year]['precip'][:]
  m = tmp != 30784
  tmp[m] = 3.0*tmp[m]
  fpo['prcp'][:,:,0,0]  = tmp[:]

  #pres
  fpo.createVariable('pres','f4',('time','ptid','latitude','longitude'))
  fpo['pres'].long_name = "Pressure"
  fpo['pres'].missing_value = 30784.0
  fpo['pres'].standard_name = "air_pressure"
  fpo['pres'].units = "Pa"
  fpo['pres'][:,:,0,0]  = output[year]['psurf']

  #shum
  fpo.createVariable('shum','f4',('time','ptid','latitude','longitude'))
  fpo['shum'].long_name = "Specific Humidity"
  fpo['shum'].missing_value = 30784.0
  fpo['shum'].standard_name = "specific_humidity"
  fpo['shum'].units = "1"
  fpo['shum'][:,:,0,0]  = output[year]['spfh']

  #tas
  fpo.createVariable('tas','f4',('time','ptid','latitude','longitude'))
  fpo['tas'].long_name = "Air Temperature"
  fpo['tas'].missing_value = 30784.0
  fpo['tas'].standard_name = "air_temperature"
  fpo['tas'].units = "K"
  fpo['tas'][:,:,0,0]  = output[year]['tair']

  #wind
  fpo.createVariable('wind','f4',('time','ptid','latitude','longitude'))
  fpo['wind'].long_name = "Wind Speed"
  fpo['wind'].missing_value = 30784.0
  fpo['wind'].standard_name = "wind speed"
  fpo['wind'].units = "m/s"
  fpo['wind'][:,:,0,0]  = output[year]['wind']

  #Close the file
  fpo.close()

 return

def extract_meteorology(mdir,dir,iyear,fyear):

 #Open access to the land database
 ldir = '%s/land' % dir
 file = '%s/ptiles.%s.tile1.h5' % (dir,dir.split('/')[-1])
 print file
 fp = h5py.File(file)
 for cell in fp['grid_data']:
  print cell
  lat = fp['grid_data'][cell]['metadata']['latitude'][0]
  lon = fp['grid_data'][cell]['metadata']['longitude'][0]
  print lat,lon
  #Create cell dirctory
  os.system('mkdir -p %s/%s' % (mdir,cell))
  #Extract the tiled meteorology for the cell
  output = extract_tile_meteorology(mdir,ldir,dir,iyear,fyear,cell)
  #Write the data out
  write_tile_meteorology(output,mdir,cell,lat,lon)
  #Hack to combine all the meteorology files (CAREFUL!!)
  os.system('cp %s/%s/*.nc %s/.' % (mdir,cell,mdir))

 return

#TO DO
#year info set outside
#no files defined in here
#lat/lon given per grid cell
#merge grid cell files into one

#Define the parameters
undef = -9999.0

#Read in the metadata
metadata = pickle.load(open(mdfile))
dir = metadata['dir']
size = metadata['npes']
iyear = metadata['meteorology']['iyear']#2002
fyear = metadata['meteorology']['fyear']#2014

#Create meteorology directory
mdir = '%s/meteorology' % dir
os.system('mkdir -p %s' % mdir)
os.system('mkdir -p %s/workspace' % mdir)

#Create the meteorology mapping
print("Creating the meteorology mapping")
create_meteorology_mapping(mdir)

#Create the data for the grid cell
print("Creating meteorology data")
extract_meteorology(mdir,dir,iyear,fyear)


#import netCDF4 as nc
import numpy as np
import geospatialtools.gdal_tools as gdal_tools
slope_exp             = 0.0
gw_res_time = 60.*86400
gw_hillslope_relief   =  100.
gw_hillslope_zeta_bar =    0.5
gw_scale_length       = 1.0
gw_scale_relief       = 1.0
gw_scale_soil_depth   = 1.0
gw_scale_perm         = 1.0

def Extract_Geohydrology_Properties_Old(lat,lon,metadata):

 workspace = 'workspace'
 #Read in the metadata
 md = gdal_tools.retrieve_metadata('workspace/perm_region.tif')
 
 #Make the lats/lons
 lats = np.linspace(md['miny']-md['resy']/2,md['maxy']+md['resy']/2,md['ny'])
 lons = np.linspace(md['minx']+md['resx']/2,md['maxx']-md['resx']/2,md['nx'])

 #Change lon
 if lon < 0: lon = lon + 360

 #Find the match
 ilat = np.argmin(np.abs(lats-lat))
 ilon = np.argmin(np.abs(lons-lon))

 #Extract and assign the geohydrology information
 output = {}

 if metadata['gw'] == 'hill_ar5':

  #hillslope_length
  var = 'hillslope_length'
  hillslope_length = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = gw_scale_length*hillslope_length
  #slope
  var = 'slope'
  slope = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = slope
  #hillslope_relief
  var = 'hillslope_relief'
  output[var] = gw_scale_relief*slope*hillslope_length
  #soil_e_depth
  var = 'soil_e_depth'
  soil_e_depth = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = gw_scale_soil_depth*soil_e_depth
  #perm
  var = 'perm'
  perm = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = perm
  #hillslope_a
  var = 'hillslope_a'
  hillslope_a = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = hillslope_a
  #hillslope_n
  var = 'hillslope_n'
  hillslope_n = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = hillslope_n

 elif metadata['gw'] == 'hill':

  #hillslope_length
  var = 'hillslope_length'
  hillslope_length = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = gw_scale_length*hillslope_length
  #slope
  var = 'slope'
  slope = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = slope
  #hillslope_relief
  var = 'gw_hillslope_relief'
  output[var] = gw_scale_relief*slope*hillslope_length
  #soil_e_depth
  var = 'gw_soil_e_depth'
  soil_e_depth = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = gw_scale_soil_depth*soil_e_depth
  #perm
  var = 'gw_perm'
  perm = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = perm
  #hillslope_a
  var = 'gw_hillslope_a'
  hillslope_a = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = hillslope_a
  #hillslope_n
  var = 'gw_hillslope_n'
  hillslope_n = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output[var] = hillslope_n

 elif metadata['gw'] == 'tiled':

  #hillslope_length
  var = 'hillslope_length'
  hillslope_length = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output['gw_hillslope_length'] = hillslope_length
  #slope
  var = 'slope'
  slope = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  #hillslope_relief
  var = 'gw_hillslope_relief'
  output[var] = slope*hillslope_length
  #soil_e_depth
  var = 'soil_e_depth'
  soil_e_depth = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output['gw_soil_e_depth'] = gw_scale_soil_depth*soil_e_depth
  #perm
  var = 'perm'
  perm = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output['gw_perm'] = perm
  #hillslope_zeta_bar
  var = 'hillslope_zeta_bar'
  hillslope_zeta_bar = gdal_tools.read_raster('%s/%s_region.tif' % (workspace,var))[ilat,ilon]
  output['gw_hillslope_zeta_bar'] = hillslope_zeta_bar

 #Other
 output['gw_res_time'] = gw_res_time
 #output['gw_hillslope_relief'] = gw_hillslope_relief
 #output['gw_hillslope_zeta_bar'] = gw_hillslope_zeta_bar
 output['gw_scale_length'] = gw_scale_length
 output['gw_scale_relief'] = gw_scale_relief
 output['gw_scale_soil_depth'] = gw_scale_soil_depth
 output['gw_scale_perm'] = gw_scale_perm
  
 return output

def Extract_Geohydrology_Properties(cdir,metadata):

 if metadata['geohydrology']['type'] == 'original':
  #Read in the necessary data
  data = {}
  for var in ['perm',]:
   data[var] = gdal_tools.read_raster('%s/%s_latlon.tif' % (cdir,var))
   data[var] = np.ma.masked_array(data[var],data[var]==-9999.0)
  output = {}
  for it in xrange(metadata['fhillslope']['NN']):
   tmp = {}
   tmp['gw_soil_e_depth'] = 3.0
   tmp['gw_perm'] = np.mean(data['perm'])
   tmp['gw_res_time'] = gw_res_time
   tmp['gw_scale_length'] = gw_scale_length
   tmp['gw_scale_relief'] = gw_scale_relief
   tmp['gw_scale_soil_depth'] = gw_scale_soil_depth
   tmp['gw_scale_perm'] = gw_scale_perm
   tmp['gw_hillslope_length'] = 1000.0
   tmp['gw_hillslope_relief'] = 300.0
   tmp['gw_hillslope_zeta_bar'] = 1.0
   #Create if necessary the variable in the output
   if tmp.keys()[0] not in output:
    for var in tmp:
     output[var] = []
   #Add the variables
   for var in tmp:
    output[var].append(tmp[var])
 
 else:
  #Read in the hru map
  #hrus = gdal_tools.read_raster('%s/soil_tiles_latlon.tif' % cdir)
  hrus = gdal_tools.read_raster('%s/soil_tiles_ea.tif' % cdir)

  #Read in the necessary data
  data = {}
  for var in ['soil_e_depth','perm_glymphs','fan2013_wtd']:
   #data[var] = gdal_tools.read_raster('%s/%s_latlon.tif' % (cdir,var))
   data[var] = gdal_tools.read_raster('%s/%s_ea.tif' % (cdir,var))
   data[var] = np.ma.masked_array(data[var],data[var]==-9999.0)

  #Extract and assign the geohydrology information
  uhrus = np.unique(hrus)
  uhrus = uhrus[uhrus != -9999]
  output = {}
  for hru in uhrus:
   m = hrus == hru
   #Create the temporary dictionary
   tmp = {}
   #Mean water table depth
   wtd = np.mean(data['fan2013_wtd'][m])
   if wtd < 0: wtd = 0.1
   if wtd > 10000.0: wtd = 10000.0
   tmp['wtd'] = wtd
   #bedrock depth
   edepth = gw_scale_soil_depth*np.mean(data['soil_e_depth'][m])/100.0 #meters
   edepth = np.ma.getdata(edepth)
   if edepth == -9999: edepth = 3.0
   if edepth < 0.1: edepth = 0.1
   if edepth > 100.0: edepth = 100.0
   if np.isnan(edepth) == True:edepth = 3.0
   if np.isinf(edepth) == True:edepth = 3.0
   tmp['gw_soil_e_depth'] = edepth
   #gw_scale_soil_depth*np.mean(data['soil_e_depth'][m])/100.0 #meters
   #tmp['gw_soil_e_depth'] = gw_scale_soil_depth*np.mean(data['soil_e_depth'][m])/100.0 #meters
   #if tmp['gw_soil_e_depth'] <= 0.0:tmp['gw_soil_e_depth']=3.0
   #if np.isnan(tmp['gw_soil_e_depth']) == 1:tmp['gw_soil_e_depth']=3.0
   gw_perm = 10**np.mean(np.ma.getdata(data['perm_glymphs'][m]))
   if gw_perm > 10.0**-10.0: gw_perm = 10.0**-10.0
   if gw_perm < 10.0**-20.0: gw_perm = 10.0**-20.0
   if np.sum(np.isnan(gw_perm) == 1) > 0: gw_perm[:] = 10.0**-15.0
   tmp['gw_perm'] = gw_perm#np.mean(data['perm'][m])
   #tmp['gw_perm'] = np.mean(data['perm'][m])
   tmp['gw_res_time'] = gw_res_time
   tmp['gw_scale_length'] = gw_scale_length
   tmp['gw_scale_relief'] = gw_scale_relief
   tmp['gw_scale_soil_depth'] = gw_scale_soil_depth
   tmp['gw_scale_perm'] = gw_scale_perm
   #TMP
   tmp['gw_hillslope_length'] = 1000.0
   tmp['gw_hillslope_relief'] = 300.0
   tmp['gw_hillslope_zeta_bar'] = 1.0
   #Create if necessary the variable in the output
   if list(tmp.keys())[0] not in output:
    for var in tmp:
     output[var] = []
   #Add the variables
   for var in tmp:
    output[var].append(tmp[var])

 #Convert all to arrays
 for var in output:
  #print(var,np.unique(output[var]))
  output[var] = np.array(output[var])
  
 return output

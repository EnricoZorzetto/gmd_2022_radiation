import numpy as np
import geospatialtools.gdal_tools as gdal_tools

#Compute the weights for each variable for each month
def Extract_Meteorology_Properties(cdir,metadata,frac):

 #Read in the hru map
 tiles = gdal_tools.read_raster('%s/tiles.tif' % cdir)

 #Define global mask
 mask = tiles != -9999

 #Extract all the mask
 masks = {}
 utiles = np.unique(tiles)
 utiles = utiles[utiles != -9999]
 for utile in utiles:
  masks[utile] = tiles == utile

 #Iterate through each meteorological variable
 output = {}
 for var in ['prec','srad','vapr','wind','tavg']:
  output[var] = []
  #Iterate per month
  for month in range(1,13):
   file = '%s/workspace/%s_%02d_latlon.tif' % (cdir,var,month)
   data = gdal_tools.read_raster(file).astype(np.float32)
   if (np.mean(data) == -9999.0):
    tmp = np.ones(frac.size)
   elif (np.mean(data[data != -9999]) == 0.0):
    tmp = np.ones(frac.size)
   else:
    tmp = np.ones(frac.size)
    #gapfill
    data[data == -9999] = np.mean(data[data != -9999])
    if var == 'tavg':data = data + 273.15
    #Compute areal mean
    mean = 0
    for itile in range(frac.size):
     if itile in utiles:
      mean += frac[itile]*np.mean(data[masks[itile]])
     else:
      mean += frac[itile]*np.mean(data)
    #Iterate per tile
    #for utile in utiles:
    for itile in range(utiles.size):
     utile = int(utiles[itile])
     tmean = np.mean(data[masks[utile]])
     weight = tmean/mean
     tmp[utile] = weight
     #tmp.append(weight)
   #Add to output array
   output[var].append(tmp)
  #Convert to array
  output[var] = np.array(output[var])

 return output

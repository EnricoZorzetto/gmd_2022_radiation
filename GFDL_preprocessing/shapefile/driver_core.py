import numpy as np 
import netCDF4 as nc
from osgeo import ogr
import osgeo.osr as osr
import pickle
import json
from scipy import interpolate
import os
import geospatialtools.gdal_tools as gdal_tools
import sys
import glob
import multiprocessing as mp
import psutil

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_percent()


def calculate_land_fraction(metadata,id,cdir,log,poly):
   
    #Retrieve info
    minlat = metadata['bbox'][2]
    minlon = metadata['bbox'][0]
    maxlat = metadata['bbox'][3]
    maxlon = metadata['bbox'][1]
    fsres = metadata['fsres']
   
    #Create a temporary shapefile with the polygon
    sdir = '%s/tmp' % cdir
    os.system('rm -rf %s' % sdir)
    os.system('mkdir -p %s' % sdir)
    file_shp = '%s/grid.shp' % sdir
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(file_shp)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    #Create the layer
    layer = ds.CreateLayer("grid", srs, ogr.wkbPolygon)
    #Create the feature
    feature = ogr.Feature(layer.GetLayerDefn())
    #Set the feature geometry using the point
    feature.SetGeometry(poly)
    #Create the feature in the layer (shapefile)
    layer.CreateFeature(feature)
    #Destroy the feature to free resources
    feature.Destroy()
    #Destroy the data source to free resources
    ds.Destroy()
   
    #Create the mask
    mask_latlon = '%s/mask_latlon.tif' % (cdir,)
    os.system('rm -f %s' % mask_latlon)
    os.system('gdal_rasterize -init -9999 -l grid -a FID -te %f %f %f %f -tr %f %f -where "FID=%d" %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,0,file_shp,mask_latlon,log))
   
    #Create the land coastline map
    file_shp = metadata['mask']['files']['GSHHS']
    #'/lustre/f2/dev/Nathaniel.Chaney/data/gshhg/GSHHS_shp/f/GSHHS_f_L1.shp'
    coastline_latlon = '%s/coastline_latlon.tif' % cdir
    os.system('gdal_rasterize -init -9999 -l GSHHS_f_L1 -a level -te %f %f %f %f -tr %f %f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,coastline_latlon,log))
   
    #Create the land coastline map (antarctica)
    file_shp = metadata['mask']['files']['GSHHS_Antarctica']
    #file_shp = '/lustre/f2/dev/Nathaniel.Chaney/data/gshhg/GSHHS_shp/f/GSHHS_f_L5.shp'
    coastlineA_latlon = '%s/coastlineA_latlon.tif' % cdir
    os.system('gdal_rasterize -init -9999 -l GSHHS_f_L5 -a level -te %f %f %f %f -tr %f %f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,coastlineA_latlon,log))
   
    #Create the political boundaries map
    if metadata['political_boundaries'] == 'undefined':
        pass
    elif metadata['political_boundaries'] == 'CONUS':
        file_shp = '/lustre/f2/dev/Nathaniel.Chaney/data/conus/s_11au16.shp'
        pbon_latlon = '%s/pbon_latlot.tif' % cdir
        os.system('gdal_rasterize -init -9999 -l s_11au16 -a LON -te %f %f %f %f -tr %f %f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,pbon_latlon,log))
   
    #Read in the coastlines and masks
    cl = gdal_tools.read_raster(coastline_latlon)
    clA = gdal_tools.read_raster(coastlineA_latlon)
    mask = gdal_tools.read_raster(mask_latlon)
   
    #Combine the coastlines
    clc = np.zeros(cl.shape)
    clc[:] = -9999
    clc[cl != -9999] = 1
    clc[clA != -9999] = 1
   
    #Define the political boundaries
    if metadata['political_boundaries'] != 'undefined':
        pbon = gdal_tools.read_raster(pbon_latlon)
        clc[pbon == -9999] = -9999
   
    #Calculate land fraction
    nland = np.sum(clc[mask == 0] != -9999)
    ncells = np.sum(mask != -9999)
    lfrac = float(nland)/float(ncells)
   
    #If it is below 1% then assume no land
    if lfrac < 0.01:lfrac = 0.0
   
    return lfrac

def interp(data,type):
    
    if type == 'lon':
        len = data.size
        diff = data - data[0]
    
    f = interpolate.interp1d([0,0.5,1.0],data,kind='quadratic')
    xnew = np.linspace(0,1,npoints)
    dnew = f(xnew)
   
    return dnew

def compute_info(xs,ys,metadata,count,cdir,log):
   
    output = {}
    #Extract regional boundaries
    xs_subset = xs[jmin:jmax+1,imin:imax+1]
    #if (np.max(xs_subset) > 100) and (np.min(xs_subset) < -100):
    # xs_subset[xs_subset < 0] = xs_subset[xs_subset < 0] + 360.0
    
    #Construct list of points
    mpoint = ogr.Geometry(ogr.wkbMultiPoint)
    point = ogr.Geometry(ogr.wkbPoint)
    for i in range(imin,imax+1):
        for j in range(jmin,jmax+1):
            point.AddPoint(xs[j,i],ys[j,i])
            #point.AddPoint(ys[j,i],xs[j,i])
            mpoint.AddGeometry(point)
    poly = mpoint.ConvexHull()
   
    #Calculate land fraction
    #Retrieve coordinates of envelope
    bbox = poly.GetEnvelope()
    metadata['bbox'] = bbox
    #Calculate the original land fraction
    lfrac_org = lm[jcell,icell]
    #Calculate the land fraction
    if ((metadata['grid'] == 'predefined') & (lfrac_org == 0.0)):
        return
    else:
        lfrac = calculate_land_fraction(metadata,count,cdir,log,poly)
   
    #Add to the info
    # create the feature
    feature = ogr.Feature(layer.GetLayerDefn())
    #Set the calculate land fraction 
    feature.SetField("LFN",lfrac)
    #Set the land fraction from the grid spec
    feature.SetField("LFO",lfrac_org)
    #Set the ID
    feature.SetField("ID",count)
    #Set the tile
    feature.SetField("TILE",tile)
    #Set the i id
    feature.SetField("X",icell+1)
    #Set the j id
    feature.SetField("Y",jcell+1)
    # Set the feature geometry using the point
    feature.SetGeometry(poly)
    # Create the feature in the layer (shapefile)
    layer.CreateFeature(feature)
    # Destroy the feature to free resources
    feature.Destroy()
   
    return
   
mdfile = sys.argv[1]
rank = int(sys.argv[2])
size = int(sys.argv[3])
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))
npoints = 3
ntiles = metadata['ntiles']
dir = metadata['dir']
tgs = metadata['gs_template']
tlm = metadata['lm_template']

#Create a directory for the rank
cdir = '%s/shapefile/rank%d' % (dir,rank)
os.system('rm -rf %s' % cdir)
os.system('mkdir -p %s' % cdir)
log = '%s/log.txt' % cdir

#Create the shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.CreateDataSource("%s/grid.shp" % cdir)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
layer = ds.CreateLayer("grid", srs, ogr.wkbPolygon)
layer.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
layer.CreateField(ogr.FieldDefn("TILE", ogr.OFTInteger))
layer.CreateField(ogr.FieldDefn("X", ogr.OFTInteger))
layer.CreateField(ogr.FieldDefn("Y", ogr.OFTInteger))
layer.CreateField(ogr.FieldDefn("LFN", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("LFO", ogr.OFTReal))

#Iterate through each tile
count = 0
for tile in range(1,ntiles+1):
    #Extract xs,ys,and lms
    gs_tile = tgs.replace('$tid',str(tile))
    lm_tile = tlm.replace('$tid',str(tile))
    print(gs_tile)
    fp  = nc.Dataset(gs_tile)
    fplm = nc.Dataset(lm_tile)
    #Extract lats/lons
    xs = fp.variables['x'][:]
    ys = fp.variables['y'][:]
    xs[xs > 180.0] = xs[xs > 180.0] - 360.0
    #Extract mask
    lm = fplm.variables['mask'][:]
    fp.close()
    fplm.close()
   
    #Iterate through each cell
    ncells_x = int((xs.shape[1]-1)/2)
    ncells_y = int((ys.shape[0]-1)/2)
    for icell in range(ncells_x):
        imin = icell*2
        imax = (icell+1)*2
        for jcell in range(ncells_y):
            jmin = jcell*2
            jmax = (jcell+1)*2
            #Skip 87-90N and 87-90S
            minlat = np.min(ys[jmin:jmax+1,imin:imax+1])
            maxlat = np.max(ys[jmin:jmax+1,imin:imax+1])
            minlon = np.min(xs[jmin:jmax+1,imin:imax+1])
            maxlon = np.max(xs[jmin:jmax+1,imin:imax+1])
            # EZDEV
            # print('*******************')
            # print('**** shapefile - driver_core.py')
            # print('minlat = {}, maxlat = {}, minlon = {}, maxlon = {}'.format(minlat, maxlat, minlon, maxlon))
            # print('*******************')
            if ((minlat >= 86.0) | (maxlat <= -86.0)):continue #Do not deal with poles
            if ((minlon < -90) & (maxlon > 90)):continue #Do not go over dateline
            #Update the count
            count += 1
            if (count-1) % size != rank:continue
            # print(tile, minlat, maxlat, minlon, maxlon)
            #Compute the info
            compute_info(xs,ys,metadata,count,cdir,log)
      
# Destroy the data source to free resources
ds.Destroy()

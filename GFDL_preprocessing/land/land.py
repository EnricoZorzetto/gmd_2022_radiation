import numpy as np
import pickle
from soils import soil_properties
from vegetation import vegetation_properties
from geohydrology import geohydrology_properties
from hillslope import hillslope_properties
from meteorology import meteorology_properties
from lake import lake_properties
from glacier import glacier_properties
import geospatialtools.gdal_tools as gdal_tools
import geospatialtools.pedotransfer as pedotransfer
import geospatialtools.terrain_tools as terrain_tools
import os
import h5py
import psutil
import netCDF4 as nc
import multiprocessing as mp
from topocalc.viewf import viewf, gradient_d8 # EZDEV
from scipy import signal # EZDEV
# from hillslope.ezdev import stdv_elev
#import matplotlib.pyplot as plt

eaproj = '+proj=moll +lon_0=%.16f +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs'
# buffer = 100
# buffer = 400  # EZDEV - Increase to 200 for computing tcf / svf


def determine_nid():
    nid = os.popen('uname -n').read()[0:-1]
    return nid

def rp(target,args):
    p = mp.Process(target=target,args=args)
    p.start()
    p.join()
    return

def memory_usage():
    pmemory = psutil.Process(os.getpid()).memory_percent()
    nmemory = psutil.virtual_memory().percent
    return '(Process: %s percent, Total: %s percent)' % (pmemory,nmemory)

def create_mask_buffered(metadata,id,cdir,log,buffer):
   
    #Retrieve info
    buffer = metadata['hillslope']['land_buffer']
    fsres = metadata['fsres']
    eares = metadata['fsres_meters']
    ldir = metadata['ldir']
    minlat = metadata['bbox'][2] - buffer*fsres
    minlon = metadata['bbox'][0] - buffer*fsres
    maxlat = metadata['bbox'][3] + buffer*fsres
    maxlon = metadata['bbox'][1] + buffer*fsres
   
    #Create the mask
    #file_shp = '%s/shapefile/grid.shp' % metadata['dir']
    mask_latlon = '%s/mask_buffered_latlon.tif' % (cdir,)
    #os.system('gdal_rasterize -init -9999 -l grid -a ID -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -where "ID=%d" %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,id,file_shp,mask_latlon,log))
   
    #Create the land coastline map
    file_shp = metadata['mask']['files']['GSHHS']#'/lustre/f2/dev/Nathaniel.Chaney/data/gshhg/GSHHS_shp/f/GSHHS_f_L1.shp'
    coastline_latlon = '%s/coastline_latlon.tif' % cdir
    os.system('rm -f %s' % coastline_latlon)
    os.system('gdal_rasterize -init -9999 -l GSHHS_f_L1 -a level -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,coastline_latlon,log))
   
    #Create the land coastline map (antarctica)
    file_shp = metadata['mask']['files']['GSHHS_Antarctica']#'/lustre/f2/dev/Nathaniel.Chaney/data/gshhg/GSHHS_shp/f/GSHHS_f_L5.shp'
    coastlineA_latlon = '%s/coastlineA_latlon.tif' % cdir
    os.system('rm -f %s' % coastlineA_latlon)
    os.system('gdal_rasterize -init -9999 -l GSHHS_f_L5 -a level -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,coastlineA_latlon,log))
   
    #Create the political boundaries map
    if metadata['political_boundaries'] == 'CONUS':
        file_shp = '/lustre/f2/dev/Nathaniel.Chaney/data/conus/s_11au16.shp'
        pbon_latlon = '%s/pbon_latlon.tif' % cdir
        os.system('rm -f %s' % pbon_latlon)
        os.system('gdal_rasterize -init -9999 -l s_11au16 -a LON -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,pbon_latlon,log))
        pbon = gdal_tools.read_raster(pbon_latlon)
   
    #Produce the final mask
    cl = gdal_tools.read_raster(coastline_latlon)
    clA = gdal_tools.read_raster(coastlineA_latlon)
    #mask = gdal_tools.read_raster(mask_latlon)
    tmp = np.zeros(cl.shape)
    tmp[:] = 0
    tmp[cl != -9999] = 1
    tmp[clA != -9999] = 1
    #tmp[mask == -9999] = -9999
   
    #Define the political boundaries
    if metadata['political_boundaries'] != 'undefined': tmp[pbon == -9999] = 0
   
    #Write out the mask
    md = gdal_tools.retrieve_metadata(coastline_latlon)
    md['nodata'] = -9999
    gdal_tools.write_raster(mask_latlon,md,tmp)
   
    #Retrieve the unbuffered mask metadata
    md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)
    md['nodata'] = -9999
    minx = md['minx'] - buffer*eares
    miny = md['miny'] - buffer*eares
    maxx = md['maxx'] + buffer*eares
    maxy = md['maxy'] + buffer*eares
   
    #Project to equal area grid
    lproj = eaproj % float((maxlon+minlon)/2)
    mask_ea = '%s/mask_buffered_ea.tif' % (cdir,)
    os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (minx,miny,maxx,maxy,eares,eares,lproj,mask_latlon,mask_ea,log))
   
    return

def create_mask(metadata,id,cdir,log):
   
    #Retrieve info
    minlat = metadata['bbox'][2]
    minlon = metadata['bbox'][0]
    maxlat = metadata['bbox'][3]
    maxlon = metadata['bbox'][1]
    fsres = metadata['fsres']
    eares = metadata['fsres_meters']
    ldir = metadata['ldir']
   
    #Create the mask
    file_shp = '%s/shapefile/grid.shp' % metadata['dir']
    mask_latlon = '%s/mask_latlon.tif' % (cdir,)
    os.system('rm -f %s' % mask_latlon)
    os.system('gdal_rasterize -init -9999 -l grid -a ID -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -where "ID=%d" %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,id,file_shp,mask_latlon,log))
   
    #Create the land coastline map
    #file_shp = '/lustre/f2/dev/Nathaniel.Chaney/data/gshhg/GSHHS_shp/f/GSHHS_f_L1.shp'
    file_shp = metadata['mask']['files']['GSHHS']
    coastline_latlon = '%s/coastline_latlon.tif' % cdir
    os.system('rm -f %s' % coastline_latlon)
    os.system('gdal_rasterize -init -9999 -l GSHHS_f_L1 -a level -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,coastline_latlon,log))
   
    #Create the land coastline map (antarctica)
    #file_shp = '/lustre/f2/dev/Nathaniel.Chaney/data/gshhg/GSHHS_shp/f/GSHHS_f_L5.shp'
    file_shp = metadata['mask']['files']['GSHHS_Antarctica']
    coastlineA_latlon = '%s/coastlineA_latlon.tif' % cdir
    os.system('rm -f %s' % coastlineA_latlon)
    os.system('gdal_rasterize -init -9999 -l GSHHS_f_L5 -a level -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,coastlineA_latlon,log))
   
    #Create the political boundaries map
    if metadata['political_boundaries'] == 'CONUS':
        file_shp = '/lustre/f2/dev/Nathaniel.Chaney/data/conus/s_11au16.shp'
        pbon_latlon = '%s/pbon_latlon.tif' % cdir
        os.system('rm -f %s' % pbon_latlon)
        os.system('gdal_rasterize -init -9999 -l s_11au16 -a LON -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp,pbon_latlon,log))
        pbon = gdal_tools.read_raster(pbon_latlon)
   
    #Produce the final mask 
    cl = gdal_tools.read_raster(coastline_latlon)
    clA = gdal_tools.read_raster(coastlineA_latlon)
    mask = gdal_tools.read_raster(mask_latlon)
    tmp = np.zeros(cl.shape)
    tmp[:] = 0
    tmp[cl != -9999] = 1
    tmp[clA != -9999] = 1
    tmp[mask == -9999] = -9999
   
    #Define the political boundaries
    if metadata['political_boundaries'] != 'undefined': tmp[pbon == -9999] = 0
   
    #Produce the final mask
    #cl = gdal_tools.read_raster(coastline_latlon)
    #clA = gdal_tools.read_raster(coastlineA_latlon)
    #mask = gdal_tools.read_raster(mask_latlon)
    #tmp = np.zeros(cl.shape)
    #tmp[:] = 0
    #tmp[cl != -9999] = 1
    #tmp[clA != -9999] = 1
    #tmp[mask == -9999] = -9999
   
    #Write out the mask
    md = gdal_tools.retrieve_metadata(mask_latlon)
    md['nodata'] = -9999
    gdal_tools.write_raster(mask_latlon,md,tmp)
   
    #Project to equal area grid
    lproj = eaproj % float((maxlon+minlon)/2)
    mask_ea = '%s/mask_ea.tif' % (cdir,)
    os.system('gdalwarp -dstnodata -9999 -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (eares,eares,lproj,mask_latlon,mask_ea,log))
   
    return

def create_terrain_products(metadata,id,cdir,workspace,log,nid,cid):
   
    #Select the database
    if metadata['topography']['type'] == 'ned':
        dem_fine = '/lustre/f2/dev/Nathaniel.Chaney/data/NED/NED.vrt'
        dem_coarse = '/lustre/f2/dev/Nathaniel.Chaney/data/NED/NED.vrt'
    elif metadata['topography']['type'] == 'gmted2010':
        dem_fine = '/lustre/f2/dev/Nathaniel.Chaney/data/gmted2010_mn75/mn75_grd'
        dem_coarse = '/lustre/f2/dev/Nathaniel.Chaney/data/gmted2010_mn30/mn30_grd'
    elif metadata['topography']['type'] == 'srtm':
        dem_fine = metadata['topography']['files']['dem_fine']#/lustre/f2/dev/Nathaniel.Chaney/data/srtm/srtm.vrt'
        dem_medium = metadata['topography']['files']['dem_medium']#/lustre/f2/dev/Nathaniel.Chaney/data/alos3d/alos3d.vrt'
        dem_coarse = metadata['topography']['files']['dem_coarse']#'/lustre/f2/dev/Nathaniel.Chaney/data/gmted/gmted2010_mn30/mn30_grd'
    else:
        print("Unknown elevation database -> Cannot process")
        exit()
   
    #Define metadata
    buffer = metadata['hillslope']['land_buffer']
    eares = metadata['fsres_meters']
    res = metadata['fsres']
    #dem_fine = metadata['dem_fine']
    #dem_coarse = metadata['dem_coarse']
    minlat = metadata['bbox'][2]
    minlon = metadata['bbox'][0]
    maxlat = metadata['bbox'][3]
    maxlon = metadata['bbox'][1]
   
    #Define boundaries of buffer
    minlon_bf = minlon - buffer*res
    minlat_bf = minlat - buffer*res
    maxlon_bf = maxlon + buffer*res
    maxlat_bf = maxlat + buffer*res
   
    #Retrieve mask metadata
    #md = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % cdir)
    md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)
    md['nodata'] = -9999
   
    #1. Cutout the region of interest
    dem_coarse_tif = '%s/dem_coarse.tif' % workspace
    dem_medium_tif = '%s/dem_medium.tif' % workspace
    dem_fine_tif = '%s/dem_fine.tif' % workspace
# EZDEV change -r average to -r lanczos? not for now - add option in the expfile
    os.system('gdalwarp -r average -dstnodata -9999 -tr %.16f %.16f -te '
            '%.16f %.16f %.16f %.16f %s %s >& %s' % (res,res,
            minlon_bf,minlat_bf,maxlon_bf,maxlat_bf,dem_fine,dem_fine_tif,log))
    os.system('gdalwarp -r average -srcnodata -9999 -dstnodata -9999 -tr '
            '%.16f %.16f -te %.16f %.16f %.16f %.16f %s %s >& %s' % (res,res,
            minlon_bf,minlat_bf,maxlon_bf,maxlat_bf,dem_medium,dem_medium_tif,log))
    os.system('gdalwarp -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f '
            '%.16f %.16f %s %s >& %s' % (res,res,
            minlon_bf,minlat_bf,maxlon_bf,maxlat_bf,dem_coarse,dem_coarse_tif,log))
   
    #Fill in the fine tif with the coarse one
    dem_latlon_tif = '%s/dem_latlon.tif' % cdir
    dc = gdal_tools.read_raster(dem_coarse_tif).astype(np.float32)
    dm = gdal_tools.read_raster(dem_medium_tif).astype(np.float32)
    df = gdal_tools.read_raster(dem_fine_tif).astype(np.float32)
    mdem = (df == 0.0) | (df == -9999)
    df[mdem] = dm[mdem]
    mdem = (df == 0.0) | (df == -9999)
    df[mdem] = dc[mdem]
    md2 = gdal_tools.retrieve_metadata(dem_fine_tif)
    md2['nodata'] = -9999
    gdal_tools.write_raster(dem_latlon_tif,md2,df)



    del dc,dm,df


   
   
    #Define the boundaries of the equal area with the buffer
    minx = md['minx'] - buffer*eares
    miny = md['miny'] - buffer*eares
    maxx = md['maxx'] + buffer*eares
    maxy = md['maxy'] + buffer*eares
   
    #Reproject the dem
    dem_ea_tif = '%s/dem_ea.tif' % cdir
    lproj = eaproj % float((maxlon+minlon)/2)
    os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (minx,miny,maxx,maxy,eares,eares,lproj,dem_latlon_tif,dem_ea_tif,log))
    # EZDEV changed -r average to -r lanczos - not for now

    ### EZDEV: added this to compute some properties from latlon terrain map
    topo3d_from_latlon = False
    if topo3d_from_latlon:
        svf, tcf, sinscosa, sinssina = ezdev_compute_latlon_topo3dterrain(dem_latlon_tif,
                                      cdir,minx,miny,maxx,maxy,eares,lproj, log)

    ### end topo3d
   
    #Mask out the ocean
    #dem = gdal_tools.read_raster(dem_latlon_tif).astype(np.float32)
    #md3 = gdal_tools.retrieve_metadata(dem_latlon_tif)
    #md3['nodata'] = -9999.0
    #mask = gdal_tools.read_raster('%s/mask_buffered_latlon.tif' % cdir)
    #dem[mask == -9999.0] = -9999.0
    dem = gdal_tools.read_raster(dem_ea_tif)
    md3 = gdal_tools.retrieve_metadata(dem_ea_tif)
    md3['nodata'] = -9999.0
    mask = gdal_tools.read_raster('%s/mask_buffered_ea.tif' % cdir)
    dem[mask == -9999.0] = -9999.0
   
    #Ensure all points are at or above sea level
    dem[(dem != -9999.0) & (dem < 0.0)] = 0.0
   
    #Write the data
    gdal_tools.write_raster(dem_ea_tif,md3,dem)
    #gdal_tools.write_raster(dem_latlon_tif,md3,dem)
    del dem
   
    #Read in data again
    rdem = gdal_tools.read_data(dem_ea_tif)
    #rdem = gdal_tools.read_data(dem_latlon_tif)
   
    #Calculate dx,dy,area
    #rdem = terrain_tools.calculate_area(rdem)
   
    #Define average spatial resolution (meters)
    #res = np.mean(rdem.area**0.5)
   
    #Remove pits in dem
    print(nid,cid,'TA: Removing pits', memory_usage())
    #demns_latlon = '%s/demns_latlon.tif' % cdir
    #os.system('./bin/pitremove -z %s -fel %s >& %s' % (dem_latlon_tif,demns_latlon,log))
    #demns = gdal_tools.read_data(demns_latlon).data
    #Note: Need to implement planchon
    demns = terrain_tools.ttf.remove_pits_planchon(rdem.data,eares)

    demns_topo3d = rdem.data.copy()
    print(nid,cid,'TA: Calculating slope and aspect', memory_usage())
    res_array = np.copy(demns)
    res_array[:] = eares
    # (slope,aspect) = terrain_tools.ttf.calculate_slope_and_aspect(demns_topo3d,res_array,res_array) # EZDEV commented


    # EZDEV: compute also slope and aspect in here
    # with my other 3d quantities of interest 
    slope, aspect, svf, tcf, sinscosa, sinssina, coss, radstelev, radavelev = compute_topo3d_properties(demns, eares, topo3d_from_latlon)
    
    #Compute accumulated area
    m2 = np.copy(mask)
    m2[:] = 1
    print(nid,cid,'TA: Calculating accumulated area', memory_usage())
    (area,fdir) = terrain_tools.ttf.calculate_d8_acc(demns,m2,eares)
   
    #Calculate channel initiation points (2 parameters)
    C = area/eares*slope**2
    ipoints = ((C > 200) & (area > 10**5)).astype(np.int32)
    ipoints[ipoints == 0] = -9999
   
    #Create area for channel delineation
    (ac,fdc) = terrain_tools.ttf.calculate_d8_acc_wipoints(demns,m2,ipoints,eares)
    ac[ac != 0] = area[ac != 0]
   
    #Compute the channels
    print(nid,cid,"TA: Defining channels", memory_usage())
    #ct = metadata['hillslope']['channel_threshold']
    # channels = terrain_tools.ttf.calculate_channels_wocean(ac,10**4,10**4,fdc,m2)

    # EZDEV: change thresold for defining hillslopes
    ct = metadata['hillslope']['channel_threshold']
    channels = terrain_tools.ttf.calculate_channels_wocean(ac,ct,ct,fdc,m2)
   
    #If the dem is undefined then set to undefined
    channels[rdem.data == -9999] = -9999
   
    #Compute the basins
    print(nid,cid,"TA: Defining basins",memory_usage())
    basins = terrain_tools.ttf.delineate_basins(channels,m2,fdir)
   
    #Define the hillslopes
    print(nid,cid,"TA: Defining hillslopes",memory_usage())
    hillslopes = terrain_tools.ttf.delineate_hillslopes(channels,area,fdir,m2)
   
    #Calculate the height above nearest drainage area
    print(nid,cid,"TA: Computing height above nearest drainage area",memory_usage())
    hand = terrain_tools.ttf.calculate_depth2channel(channels,basins,fdir,demns)
   
    #Calculate topographic index
    print(nid,cid,"TA: Computing topographic index",memory_usage())
    ti = np.copy(area)
    m = (area != -9999) & (slope != -9999) & (slope != 0.0)
    ti[m] = np.log(area[m]/res/slope[m])
    ti[slope == 0] = 15.0
    #ti = terrain_tools.calculate_topographic_index(area,res,slope)
   
    #Cleanup
    slope[mask != 1] = -9999
    aspect[mask != 1] = -9999
    area[mask != 1] = -9999
    channels[mask != 1] = -9999
    basins[mask != 1] = -9999
    hillslopes[mask != 1] = -9999
    hand[mask != 1] = -9999
    ti[mask != 1] = -9999
   
    svf[mask != 1] = -9999 # EZDEV
    tcf[mask != 1] = -9999 # EZDEV
    coss[mask != 1] = -9999 # EZDEV # remove? no need for clustering
    sinscosa[mask != 1] = -9999 # EZDEV
    sinssina[mask != 1] = -9999 # EZDEV
    radstelev[mask != 1] = -9999 # EZDEV
    radavelev[mask != 1] = -9999 # EZDEV

    #Cut out only region of interest
    ilats = np.arange(buffer-1,hillslopes.shape[0]-buffer)
    ilons = np.arange(buffer-1,hillslopes.shape[1]-buffer)
    hillslopes = hillslopes[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    channels = channels[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    hand = hand[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    slope = slope[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    aspect = aspect[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    demns = demns[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    basins = basins[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    ti = ti[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    area = area[ilats[0]:ilats[-1],ilons[0]:ilons[-1]]
    svf = svf[ilats[0]:ilats[-1],ilons[0]:ilons[-1]] # EZDEV
    tcf = tcf[ilats[0]:ilats[-1],ilons[0]:ilons[-1]] # EZDEV
    coss = coss[ilats[0]:ilats[-1],ilons[0]:ilons[-1]] # EZDEV # remove? no need for clustering
    sinscosa = sinscosa[ilats[0]:ilats[-1],ilons[0]:ilons[-1]] # EZDEV
    sinssina = sinssina[ilats[0]:ilats[-1],ilons[0]:ilons[-1]] # EZDEV
    radstelev = radstelev[ilats[0]:ilats[-1],ilons[0]:ilons[-1]] # EZDEV
    radavelev = radavelev[ilats[0]:ilats[-1],ilons[0]:ilons[-1]] # EZDEV

    print('EZDEV: print some statistics after cropping the resulting maps')
    print('svf:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(svf), np.mean(svf), np.std(svf)))
    print('tcf:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(tcf), np.mean(tcf), np.std(tcf)))
    print('coss:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(coss), np.mean(coss), np.std(coss)))
    print('sinscosa:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(sinscosa), np.mean(sinscosa), np.std(sinscosa)))
    print('sinssina:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(sinssina), np.mean(sinssina), np.std(sinssina)))
    print('radstelev:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(radstelev), np.mean(radstelev), np.std(radstelev)))
    print('hillslopes = {}'.format(hillslopes))
    # END EZDEV
   
    #Output all the terrain variables
    output = {
        'tcf':tcf, # EZDEV
        'svf':svf, # EZDEV
        'coss': coss,  # EZDEV # remove? no need for clustering
        'sinscosa': sinscosa,  # EZDEV
        'sinssina': sinssina,  # EZDEV
        'radstelev': radstelev,  # EZDEV
        'radavelev': radavelev,  # EZDEV
              'demns':demns,
              'slope':slope,
              'aspect':aspect,
              'area':area,
              'channels':channels,
              'basins':basins,
              'hillslopes':hillslopes,
              'hand':hand,
              'ti':ti
             }
    for var in output:
     #ofile = '%s/%s_latlon.tif' % (cdir,var)
     ofile = '%s/%s_ea.tif' % (cdir,var)
     gdal_tools.write_raster(ofile,md,output[var])
   
    return

def create_lake_products(metadata,cdir,workspace,log):
   
    #Define metadata
    minlat = metadata['bbox'][2]
    minlon = metadata['bbox'][0]
    maxlat = metadata['bbox'][3]
    maxlon = metadata['bbox'][1]
    fsres = metadata['fsres']
    eares = metadata['fsres_meters']
   
    #Read in the lakes
    file_shp = metadata['lake']['file']#'/lustre/f2/dev/Nathaniel.Chaney/data/hydrolakes/HydroLAKES_polys_v10_shp'
    #Subset
    file_shp_ss = '%s/lakes_latlon.shp' % cdir
    os.system('ogr2ogr -spat %.16f %.16f %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,file_shp_ss,file_shp,log))
    #Rasterize
    lakes_latlon = '%s/lakes_latlon.tif' % (cdir,)
    os.system('gdal_rasterize -init -9999 -l lakes_latlon -a Lake_area -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp_ss,lakes_latlon,log))
    lakes = gdal_tools.read_raster(lakes_latlon)
    lakes[lakes != -9999] = 1
   
    #3. Read in the mask
    file_mask = '%s/mask_latlon.tif' % (cdir,)
    mask = gdal_tools.read_raster(file_mask)
    lakes[mask != 1] = -9999
    
    #Output the combined map
    #lakes_latlon = '%s/lakes_latlon.tif' % (cdir,)
    md = gdal_tools.retrieve_metadata(lakes_latlon)
    md['nodata'] = -9999
    gdal_tools.write_raster(lakes_latlon,md,lakes)
   
    #2. Project to equal area grid
    lproj = eaproj % float((maxlon+minlon)/2)
    lakes_ea = '%s/lakes_ea.tif' % (cdir,)
    os.system('gdalwarp -dstnodata -9999 -tr %.16f %.16f -s_srs EPSG:4326 -t_srs "%s" %s %s >& %s' % (eares,eares,lproj,lakes_latlon,lakes_ea,log))
   
    return

def create_glacier_products(metadata,cdir,workspace,log):
   
    #Define metadata
    minlat = metadata['bbox'][2]
    minlon = metadata['bbox'][0]
    maxlat = metadata['bbox'][3]
    maxlon = metadata['bbox'][1]
    fsres = metadata['fsres']
    eares = metadata['fsres_meters']
   
    #1. Create lat/lon rasters for region (glims)
    file_shp = metadata['glacier']['files']['glims']#'/lustre/f2/dev/Nathaniel.Chaney/data/glims_db_20150728/glims_polygons.shp'
    #Subset
    file_shp_ss = '%s/glaciers_latlon.shp' % cdir
    os.system('ogr2ogr -spat %.16f %.16f %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,file_shp_ss,file_shp,log))
    #Rasterize
    glaciers_latlon = '%s/glaciers_latlon.tif' % (cdir,)
    os.system('gdal_rasterize -init -9999 -l glaciers_latlon -a glac_id -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_shp_ss,glaciers_latlon,log))
   
    #2. Create raster from glcc
    file_glcc = metadata['glacier']['files']['glcc']#'/lustre/f2/dev/Nathaniel.Chaney/data/glcc/gbogegeo20.tif'
    glcc_latlon_tif = '%s/glaciers_glcc_latlon.tif' % workspace
    os.system('gdalwarp -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f %s %s >& %s' % (fsres,fsres,minlon,minlat,maxlon,maxlat,file_glcc,glcc_latlon_tif,log))
   
    #3. Read in the mask
    file_mask = '%s/mask_latlon.tif' % (cdir,)
    mask = gdal_tools.read_raster(file_mask)
   
    #Combine the two glaciers maps
    glcc = gdal_tools.read_raster(glcc_latlon_tif)
    glims = gdal_tools.read_raster(glaciers_latlon)
    lat = (minlat+maxlat)/2
    glims[glims != -9999] = 1
    if (lat > 60) or (lat < -60):
        glims[glcc == 12] = 1
    #Mask out
    glims[mask != 1] = -9999
    md = gdal_tools.retrieve_metadata(glaciers_latlon)
    md['nodata'] = -9999
    gdal_tools.write_raster(glaciers_latlon,md,glims)
   
    #2. Project to equal area grid
    lproj = eaproj % float((maxlon+minlon)/2)
    glaciers_ea = '%s/glaciers_ea.tif' % (cdir,)
    os.system('gdalwarp -dstnodata -9999 -tr %.16f %.16f -s_srs EPSG:4326 -t_srs "%s" %s %s >& %s' % (eares,eares,lproj,glaciers_latlon,glaciers_ea,log))
   
    return

def create_geohydrology_products(metadata,cdir,workspace,log):

    #Define metadata
    minlat = metadata['bbox'][2]
    minlon = metadata['bbox'][0]
    maxlat = metadata['bbox'][3]
    maxlon = metadata['bbox'][1]
    fsres = metadata['fsres']
    eares = metadata['fsres_meters']
    #dir = '/lustre/f2/dev/Nathaniel.Chaney/data/geohydrology'
    dir = metadata['geohydrology']['dir']
   
    #Retrieve mask metadata
    md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)
   
    #1. Cutout the region of interest
    #for var in ['perm_glymphs','soil_e_depth','fan2013_wtd']:
    for var in ['perm_glymphs','uhst_p2016','uhrt_p2016','lt_uvt_p2016','ul_mask_p2016','fan2013_wtd','soil_e_depth']:
        file_in = '%s/%s.tif' % (dir,var)
        file_latlon = '%s/%s_latlon.tif' % (cdir,var)
        if var in ['uhrt_p2016','lt_uvt_p2016','ul_mask_p2016']:
            os.system('gdalwarp -t_srs EPSG:4326 -srcnodata 255 -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f %s %s >& %s' % (fsres,fsres,minlon,minlat,maxlon,maxlat,file_in,file_latlon,log))
        elif var in ['uhst_p2016',]:
            os.system('gdalwarp -t_srs EPSG:4326 -srcnodata -1 -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f %s %s >& %s' % (fsres,fsres,minlon,minlat,maxlon,maxlat,file_in,file_latlon,log))
        else:
            os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f %s %s >& %s' % (fsres,fsres,minlon,minlat,maxlon,maxlat,file_in,file_latlon,log))
        file_ea = '%s/%s_ea.tif' % (cdir,var)
        lproj = eaproj % float((maxlon+minlon)/2)
        os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
   
    return

def create_watermanagement_products(metadata,cdir,workspace,log):
   
    #Define metadata
    minlat = metadata['bbox'][2]
    minlon = metadata['bbox'][0]
    maxlat = metadata['bbox'][3]
    maxlon = metadata['bbox'][1]
    fsres = metadata['fsres']
    eares = metadata['fsres_meters']
   
    #Retrieve mask metadata
    md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)
   
    #1. Cutout the region of interest
    var = 'irrigation'
    file_in = metadata['irrigation']['file']
    #file_in = '/lustre/f2/dev/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3'
    file_latlon = '%s/%s_latlon.tif' % (cdir,var) 
    os.system('gdalwarp -s_srs "+proj=laea +lat_0=45.0 +lon_0=-100.0 +x_0=0 +y_0=0 +a=6370997.0 +b=6370997.0 +units=m +no_defs" -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_in,file_latlon,log))
    file_ea = '%s/%s_ea.tif' % (cdir,var)
    lproj = eaproj % float((maxlon+minlon)/2)
    os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
   
    return

def create_landcover_products(metadata,cdir,workspace,log):
    
    if metadata['landcover']['type'] == 'cld':
        create_landcover_products_cld(metadata,cdir,workspace,log)
    elif metadata['landcover']['type'] == 'globcover2009':
        create_landcover_products_globcover2009(metadata,cdir,workspace,log)
    elif metadata['landcover']['type'] == 'cci':
        create_landcover_products_cci(metadata,cdir,workspace,log)
    else:
        print("Unknown land cover database -> Cannot process")
        exit()
     
    return

def create_landcover_products_cci(metadata,cdir,workspace,log):

 #Define metadata
 minlat = metadata['bbox'][2]
 minlon = metadata['bbox'][0]
 maxlat = metadata['bbox'][3]
 maxlon = metadata['bbox'][1]
 fsres = metadata['fsres']
 eares = metadata['fsres_meters']

 #Retrieve mask metadata
 #md = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % cdir)
 md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)

 #1. Cutout the region of interest
 #c3/c4 distribution
 #files = ['/lustre/f2/dev/Nathaniel.Chaney/data/LUH2/tiff/crop\$c3.tif','/lustre/f2/dev/Nathaniel.Chaney/data/LUH2/tiff/crop\$c4.tif']
 vars = {'c3':metadata['landcover']['files']['c3'],#'/lustre/f2/dev/Nathaniel.Chaney/data/LUH2/tiff/crop\$c3.tif',
         'c4':metadata['landcover']['files']['c4'],#'/lustre/f2/dev/Nathaniel.Chaney/data/LUH2/tiff/crop\$c4.tif',
         'cheight':metadata['landcover']['files']['cheight'],#'/lustre/f2/dev/Nathaniel.Chaney/data/cheight/Simard_Pinto_3DGlobalVeg_JGR.tif',
         'maxlai':metadata['landcover']['files']['maxlai'],}#'/lustre/f2/dev/Nathaniel.Chaney/data/modis/mcd15a2h/stats/max.vrt'}
 tmp = {}
 for var in vars:
  file_in = vars[var]
  file_latlon = '%s/%s_latlon.tif' % (cdir,var)
  #os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon+360,minlat,maxlon+360,maxlat,fsres,fsres,file_in,file_latlon,log))
  os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_in,file_latlon,log))
  file_ea = '%s/%s_ea.tif' % (cdir,var)
  lproj = eaproj % float((maxlon+minlon)/2)
  os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
  tmp[var] = gdal_tools.read_raster(file_ea)
  #tmp[var] = gdal_tools.read_raster(file_latlon)
  if (np.unique(tmp[var]).size == 1) & (np.unique(tmp[var])[0] == -9999):
   tmp[var] = 0.0
  else:
   tmp[var] = np.mean(tmp[var][tmp[var] != -9999])
 #compute the fraction of crops that are c3 and c4
 fsum = tmp['c3']+tmp['c4']
 if fsum != 0.0:
  fc3 = tmp['c3']/fsum
  fc4 = tmp['c4']/fsum
 else:
  fc3 = 0.0
  fc4 = 0.0
 
 #land cover
 file_in = metadata['landcover']['files']['cover']#'/lustre/f2/dev/Nathaniel.Chaney/data/CCILC/product/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif'
 var = 'cover'
 file_latlon = '%s/%s_latlon.tif' % (cdir,var)
 os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_in,file_latlon,log))
 file_ea = '%s/%s_ea.tif' % (cdir,var)
 lproj = eaproj % float((maxlon+minlon)/2)
 os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
 #define files
 #veg_file_ll = '%s/vegetation_latlon.tif' % cdir
 #lu_file_ll = '%s/landuse_latlon.tif' % cdir
 #cn_file_ll = '%s/cultivated2natural_latlon.tif' % cdir
 #gt_file_ll = '%s/grass2tree_latlon.tif' % cdir
 #de_file_ll = '%s/deciduous2evergreen_latlon.tif' % cdir
 #ur_file_ll = '%s/natural2urban_latlon.tif' % cdir
 #c3c4_file_ll = '%s/c32c4_latlon.tif' % cdir
 veg_file_ll = '%s/vegetation_ea.tif' % cdir
 lu_file_ll = '%s/landuse_ea.tif' % cdir
 cn_file_ll = '%s/cultivated2natural_ea.tif' % cdir
 gt_file_ll = '%s/grass2tree_ea.tif' % cdir
 de_file_ll = '%s/deciduous2evergreen_ea.tif' % cdir
 ur_file_ll = '%s/natural2urban_ea.tif' % cdir
 c3c4_file_ll = '%s/c32c4_ea.tif' % cdir

 #SP_C4GRASS   = 0, & ! c4 grass
 #SP_C3GRASS   = 1, & ! c3 grass
 #SP_TEMPDEC   = 2, & ! temperate deciduous
 #SP_TROPICAL  = 3, & ! non-grass tropical
 #SP_EVERGR    = 4    ! non-grass evergreen
 #N_LU_TYPES = 5, & ! number of different land use types
 #LU_PAST    = 1, & ! pasture
 #LU_CROP    = 2, & ! crops
 #LU_NTRL    = 3, & ! natural vegetation
 #LU_SCND    = 4, & ! secondary vegetation
 #LU_URBN    = 5, & ! urban
 #LU_PSL     = 1001 ! primary and secondary land, for LUMIP
 #Cold deciduous set to evergreen...

 #Construct the CCILC lookup table
 table = {
          10:{'lu':2,'vg':10,'name':'Cropland rainfed'},
          11:{'lu':2,'vg':10,'name':'Herbaceous cover'},
          14:{'lu':2,'vg':10,'name':'Tree or shrub cover'},
          20:{'lu':2,'vg':10,'name':'Cropland irrigated or post-flooding'},
          30:{'lu':2,'vg':10,'name':'Mosaic cropland (>50%) / natural vegetation (tree,shrub,herbaceous cover) (<50%)'},
          40:{'lu':2,'vg':10,'name':'Mosaic natural vegetation (tree,shrub,herbaceous cover) (>50%) / cropland (<50%)'},
          50:{'lu':3,'vg':3,'name':'Tree cover,broadleaved,evergreen,closed to open (>15%)'},
          60:{'lu':3,'vg':23,'name':'Tree cover,broadleaved,deciduous,closed to open (>15%)'},
          61:{'lu':3,'vg':23,'name':'Tree cover,broadleaved,deciduous,closed (>40%)'},
          62:{'lu':3,'vg':23,'name':'Tree cover,broadleaved,deciduous,open (15-40%)'},
          70:{'lu':3,'vg':4,'name':'Tree cover,needleleaved,evergreen,closed to open (>15%)'},
          71:{'lu':3,'vg':4,'name':'Tree cover,needleleaved,evergreen,closed (>40%)'},
          72:{'lu':3,'vg':4,'name':'Tree cover,needleleaved,evergreen,open (15-40%)'},
          #80:{'lu':3,'vg':2,'name':'Tree cover,needleleaved,deciduous,closed to open (>15%)'},
          #81:{'lu':3,'vg':2,'name':'Tree cover,needleleaved,deciduous,closed (>40%)'},
          #82:{'lu':3,'vg':2,'name':'Tree cover,needleleaved,deciduous,open (15-40%)'},
          80:{'lu':3,'vg':24,'name':'Tree cover,needleleaved,deciduous,closed to open (>15%)'},
          81:{'lu':3,'vg':24,'name':'Tree cover,needleleaved,deciduous,closed (>40%)'},
          82:{'lu':3,'vg':24,'name':'Tree cover,needleleaved,deciduous,open (15-40%)'},
          90:{'lu':3,'vg':234,'name':'Tree cover,mixed leaf type (broadleaved and needleleaved)'},
          100:{'lu':3,'vg':234,'name':'Mosaic tree and shrub (>50%) / herbaceous cover (<50%)'},
          110:{'lu':3,'vg':234,'name':'Mosaic herbaceous cover (>50%) / tree and shrub (<50%)'},
          120:{'lu':3,'vg':234,'name':'Shrubland'},
          121:{'lu':3,'vg':4,'name':'Shrubland evergreen'},
          122:{'lu':3,'vg':2,'name':'Shrubland deciduous'},
          130:{'lu':3,'vg':10,'name':'Grassland'},
          140:{'lu':3,'vg':1,'name':'Lichens and mosses'},
          150:{'lu':3,'vg':234,'name':'Sparse vegetation (tree,shrub,herbaceous cover) (<15%)'},
          151:{'lu':3,'vg':234,'name':'Sparse tree (<15%)'},
          152:{'lu':3,'vg':234,'name':'Sparse shrub (<15%)'},
          153:{'lu':3,'vg':234,'name':'Sparse herbaceous cover (<15%)'},
          160:{'lu':3,'vg':234,'name':'Tree cover,flooded,fresh or brakish water'},
          170:{'lu':3,'vg':234,'name':'Tree cover,flooded,saline water'},
          180:{'lu':3,'vg':234,'name':'Shrub or herbaceous cover,flooded,fresh/saline/brakish water'},
          190:{'lu':3,'vg':10,'name':'Urban areas'},
          200:{'lu':3,'vg':10,'name':'Bare areas'},
          201:{'lu':3,'vg':10,'name':'Consolidated bare areas'},
          202:{'lu':3,'vg':10,'name':'Unconsolidated bare areas'},
          210:{'lu':3,'vg':10,'name':'Water bodies'},
          220:{'lu':3,'vg':10,'name':'Permanent snow and ice'},
         }

 #Retrieve the data
 #md = gdal_tools.retrieve_metadata(file_latlon)
 #md['nodata'] = -9999.0
 #lc_data = gdal_tools.read_raster(file_latlon)
 md = gdal_tools.retrieve_metadata(file_ea)
 md['nodata'] = -9999.0
 lc_data = gdal_tools.read_raster(file_ea)
 
 #First construct the species and landuse maps
 template = np.copy(lc_data)
 template[:] = -9999.0
 veg_data = np.copy(template)
 lu_data = np.copy(template)
 ulcs = np.unique(lc_data)
 ulcs = ulcs[ulcs > 0]
 for ulc in ulcs:
  m = lc_data == ulc
  if ulc in table:
   veg_data[m] = table[ulc]['vg']
   lu_data[m] = table[ulc]['lu']
  else:
   veg_data[m] = 2 #temperate deciduous
   lu_data[m] = 3 #natural

 #Determine if c3/c4 and determine if deciduous/evergreen
 #Retrieve the climate data
 pann = gdal_tools.read_raster('%s/pann_ea.tif' % cdir)
 tann = gdal_tools.read_raster('%s/tann_ea.tif' % cdir)
 t_cold = gdal_tools.read_raster('%s/t_cold_ea.tif' % cdir)
 ncm = gdal_tools.read_raster('%s/ncm_ea.tif' % cdir)
 #pann = gdal_tools.read_raster('%s/pann_latlon.tif' % cdir)
 #tann = gdal_tools.read_raster('%s/tann_latlon.tif' % cdir)
 #t_cold = gdal_tools.read_raster('%s/t_cold_latlon.tif' % cdir)
 #ncm = gdal_tools.read_raster('%s/ncm_latlon.tif' % cdir)

 #Correct for c3/c4 (crops)
 m = (lu_data == 2) & (veg_data == 10)
 #Randomly choose according to c3/c4 fractions
 if np.sum(m) > 0:
  idx = np.where(m)
  np.random.seed(1)
  ids = np.random.choice(np.arange(idx[0].size),size=int(fc3*np.sum(m)),replace=False)
  #Set these to c3
  veg_data[idx[0][ids],idx[1][ids]] = 1
  #Set the rest to c4
  veg_data[(lu_data == 2) & (veg_data == 10)] = 0

 #Correct for c3/c4 (LM code) (non-crops)
 m = (veg_data == 10) & (pann != -9999) & (lu_data != 2)
 # Rule based on analysis of ED global output; equations from JPC, 2/02
 temp = tann #Temperature from input climate data (deg K)
 precip = pann #Precip from input climate data (mm/yr)
 pc4=np.exp(-0.0421*(273.16+25.56-temp)-(0.000048*(273.16+25.5-temp)*precip));
 pt = np.ones(temp.shape) #Initialize to C3
 mc4 = m & (pc4 > 0.5)
 mc3 = m & (pc4 <= 0.5)
 veg_data[mc3] = 1
 veg_data[mc4] = 0
 #pt[pc4 > 0.5] = 0 #Set to C4

 #Correct for tropical/temperate (LM code)
 #23 case (know its deciduous)
 m0 = (veg_data == 23) & (t_cold != -9999)
 m1 = m0 & (t_cold > 278.16)
 veg_data[m1] = 3
 m1 = m0 & (t_cold <= 278.16)
 veg_data[m1] = 2
 #234 case (only know it is a tree)
 m0 = (veg_data == 234) & (t_cold != -9999)
 m1 = m0 & (t_cold > 278.16)
 veg_data[m1] = 3
 m1 = m0 & (t_cold <= 278.16)
 veg_data[m1] = 24
 
 #Correct for temperate deciduous/evergreen (LM code)
 m = (veg_data == 24) & (ncm != -9999)
 pe = 1.0/(1.0+((1.0/0.00144)*np.exp(-0.7491*ncm)));
 mevg = m & ((pe > 0.5) & (pe < 0.9))
 veg_data[mevg] = 4
 mdec = m & ((pe < 0.5) | (pe > 0.9))
 veg_data[mdec] = 2

 #Create the rest of products
 cn_mapping = {1:0,2:0,3:1,4:1}
 gt_mapping = {0:0,1:0,2:1,3:1,4:1}
 de_mapping = {0:0,1:0,2:0,3:1,4:1}
 c3c4_mapping = {0:1,1:0,2:0,3:0,4:0}
 cn_data = np.copy(template)
 gt_data = np.copy(template)
 de_data = np.copy(template)
 c3c4_data = np.copy(template)
 us = np.unique(veg_data)
 us = us[us != -9999]
 for u in us:
  m = veg_data == u 
  gt_data[m] = gt_mapping[int(u)]
  de_data[m] = de_mapping[int(u)]
  c3c4_data[m] = c3c4_mapping[int(u)]
 us = np.unique(lu_data)
 us = us[us != -9999]
 for u in us:
  m = lu_data == u
  cn_data[m] = cn_mapping[int(u)]

 #Write out all the files
 gdal_tools.write_raster(veg_file_ll,md,veg_data)
 gdal_tools.write_raster(lu_file_ll,md,lu_data)
 gdal_tools.write_raster(cn_file_ll,md,cn_data)
 gdal_tools.write_raster(gt_file_ll,md,gt_data)
 gdal_tools.write_raster(de_file_ll,md,de_data)
 gdal_tools.write_raster(c3c4_file_ll,md,c3c4_data)

 return

def create_landcover_products_globcover2009(metadata,cdir,workspace,log):

 file_in = '/lustre/f2/dev/Nathaniel.Chaney/data/globecover2009/GLOBCOVER_L4_200901_200912_V2.3.tif'

 #Define metadata
 minlat = metadata['bbox'][2]
 minlon = metadata['bbox'][0]
 maxlat = metadata['bbox'][3]
 maxlon = metadata['bbox'][1]
 fsres = metadata['fsres']
 eares = metadata['fsres_meters']

 #Retrieve mask metadata
 md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)

 #1. Cutout the region of interest
 var = 'cover'
 file_latlon = '%s/%s_latlon.tif' % (cdir,var) 
 os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_in,file_latlon,log))
 file_ea = '%s/%s_ea.tif' % (cdir,var)
 lproj = eaproj % float((maxlon+minlon)/2)
 os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
 veg_file_ea = '%s/vegetation_ea.tif' % cdir
 lu_file_ea = '%s/landuse_ea.tif' % cdir
 cn_file_ea = '%s/cultivated2natural_ea.tif' % cdir
 gt_file_ea = '%s/grass2tree_ea.tif' % cdir
 de_file_ea = '%s/deciduous2evergreen_ea.tif' % cdir
 ur_file_ea = '%s/natural2urban_ea.tif' % cdir

 #SP_C4GRASS   = 0, & ! c4 grass
 #SP_C3GRASS   = 1, & ! c3 grass
 #SP_TEMPDEC   = 2, & ! temperate deciduous
 #SP_TROPICAL  = 3, & ! non-grass tropical
 #SP_EVERGR    = 4    ! non-grass evergreen
 #N_LU_TYPES = 5, & ! number of different land use types
 #LU_PAST    = 1, & ! pasture
 #LU_CROP    = 2, & ! crops
 #LU_NTRL    = 3, & ! natural vegetation
 #LU_SCND    = 4, & ! secondary vegetation
 #LU_URBN    = 5, & ! urban
 #LU_PSL     = 1001 ! primary and secondary land, for LUMIP

 #Construct the Globcover2009 lookup table
 table = {
          11:{'lu':2,'vg':1,'name':'Post-flooding or irrigated croplands (or aquatic)'},
          14:{'lu':2,'vg':1,'name':'Rainfed croplands'},
          20:{'lu':2,'vg':1,'name':'Mosaic cropland (50-70%) / vegetation (grassland/shrubland/forest) (20-50%)'},
          30:{'lu':3,'vg':1,'name':'Mosaic vegetation (grassland/shrubland/forest) (50-70%) / cropland (20-50%)'},
          40:{'lu':3,'vg':4,'name':'Closed to open (>15%) broadleaved evergreen or semi-deciduous forest (>5m)'},
          50:{'lu':3,'vg':2,'name':'Closed (>40%) broadleaved deciduous forest (>5m)'},
          60:{'lu':3,'vg':2,'name':'Open (15-40%) broadleaved deciduous forest/woodland (>5m)'},
          70:{'lu':3,'vg':4,'name':'Closed (>40%) needleleaved evergreen forest (>5m)'},
          90:{'lu':3,'vg':4,'name':'Open (15-40%) needleleaved deciduous or evergreen forest (>5m)'},
          100:{'lu':3,'vg':2,'name':'Closed to open (>15%) mixed broadleaved and needleleaved forest (>5m)'},
          110:{'lu':3,'vg':4,'name':'Mosaic forest or shrubland (50-70%) / grassland (20-50%)'},         
          120:{'lu':3,'vg':1,'name':'Mosaic grassland (50-70%) / forest or shrubland (20-50%)'},
          130:{'lu':3,'vg':4,'name':'Closed to open (>15%) (broadleaved or needleleaved, evergreen or deciduous) shrubland (<5m)'},
          140:{'lu':3,'vg':1,'name':'Closed to open (>15%) herbaceous vegetation (grassland, savannas or lichens/mosses)'},
          150:{'lu':3,'vg':1,'name':'Sparse (<15%) vegetation'},
          160:{'lu':3,'vg':4,'name':'Closed to open (>15%) broadleaved forest regularly flooded (semi-permanently or temporarily) - Fresh or brackish water'},
          170:{'lu':3,'vg':4,'name':'Closed (>40%) broadleaved forest or shrubland permanently flooded - Saline or brackish water'},
          180:{'lu':3,'vg':1,'name':'Closed to open (>15%) grassland or woody vegetation on regularly flooded or waterlogged soil - Fresh, brackish or saline water'},
          190:{'lu':3,'vg':1,'name':'Artificial surfaces and associated areas (Urban areas >50%)'}, 
          200:{'lu':3,'vg':1,'name':'Bare areas'},
          210:{'lu':3,'vg':1,'name':'Water bodies'},
          220:{'lu':3,'vg':1,'name':'Permanent snow and ice'},
          230:{'lu':3,'vg':1,'name':'No data (burnt areas, clouds,...)'},
         }

 cn_mapping = {1:0,2:0,3:1,4:1}
 gt_mapping = {0:0,1:0,2:1,3:1,4:1}
 de_mapping = {0:0,1:0,2:0,3:1,4:1}
 #Retrieve the data
 md = gdal_tools.retrieve_metadata(file_ea)
 cld_data = gdal_tools.read_raster(file_ea)
 md['nodata'] = -9999.0
 #2. Construct the vegetation type and landuse map
 ulcs = np.unique(cld_data)
 ulcs = ulcs[ulcs != 0]
 veg_data = np.copy(cld_data)
 lu_data = np.copy(cld_data)
 cn_data = np.copy(cld_data)
 gt_data = np.copy(cld_data)
 de_data = np.copy(cld_data)
 ur_data = np.copy(cld_data)
 veg_data[:] = -9999.0
 lu_data[:] = -9999.0
 cn_data[:] = -9999.0
 gt_data[:] = -9999.0
 de_data[:] = -9999.0
 ur_data[:] = -9999.0
 for ulc in ulcs:
  m = cld_data == ulc
  if ulc in table:
   veg_data[m] = table[ulc]['vg']
   lu_data[m] = table[ulc]['lu']
   cn_data[m] = cn_mapping[table[ulc]['lu']]
   gt_data[m] = gt_mapping[table[ulc]['vg']]   
   de_data[m] = de_mapping[table[ulc]['vg']]
  else:  
   veg_data[m] = 2 #temperature deciduous
   lu_data[m] = 3 #natural
   cn_data[m] = 1
   gt_data[m] = 1
   de_data[m] = 0
  #if ulc in [121,122,123,124]:ur_data[m] = 1
  #else: ur_data[m] = 0
 #veg_data[veg_data == 0.0] = -9999.0
 #lu_data[veg_data == 0.0] = -9999.0
 gdal_tools.write_raster(lu_file_ea,md,lu_data)
 gdal_tools.write_raster(cn_file_ea,md,cn_data)
 gdal_tools.write_raster(gt_file_ea,md,gt_data)
 gdal_tools.write_raster(de_file_ea,md,de_data)
 gdal_tools.write_raster(ur_file_ea,md,ur_data) 

 #Retrieve the climate data
 pann = gdal_tools.read_raster('%s/pann_ea.tif' % cdir)
 tann = gdal_tools.read_raster('%s/tann_ea.tif' % cdir)
 t_cold = gdal_tools.read_raster('%s/t_cold_ea.tif' % cdir)
 ncm = gdal_tools.read_raster('%s/ncm_ea.tif' % cdir)

 #Correct for c3/c4 (LM code)
 m0 = (lu_data == 1) | (lu_data == 2) | (lu_data == 3) #Natural,crops, or pasture
 m = m0 & ((veg_data == 0) | (veg_data ==  1))
 m = m & (pann != -9999)
 # Rule based on analysis of ED global output; equations from JPC, 2/02
 temp = tann #Temperature from input climate data (deg K)
 precip = pann #Precip from input climate data (mm/yr)
 pc4=np.exp(-0.0421*(273.16+25.56-temp)-(0.000048*(273.16+25.5-temp)*precip));
 pt = np.ones(temp.shape) #Initialize to C3
 pt[pc4 > 0.5] = 0 #Set to C4
 veg_data[m] = pt[m]
 
 #Correct for deciduous/evergreen (LM code)
 m0 = ((veg_data == 2) | (veg_data == 4))
 m0 = m0 & (ncm != -9999)
 pe = 1.0/(1.0+((1.0/0.00144)*np.exp(-0.7491*ncm)));
 m = m0 & ((pe > 0.5) & (pe < 0.9))
 veg_data[m] = 4 
 m = m0 & ((pe < 0.5) | (pe > 0.9))
 veg_data[m] = 2

 #Correct for tropical (LM code)
 m = (veg_data == 2)
 m = m & (t_cold != -9999)
 m = m & (t_cold > 291.16)#278.16) (wiki)
 veg_data[m] = 3

 #Output the final vegetation
 gdal_tools.write_raster(veg_file_ea,md,veg_data)

 return

def create_landcover_products_cld(metadata,cdir,workspace,log):

 #Define metadata
 minlat = metadata['bbox'][2]
 minlon = metadata['bbox'][0]
 maxlat = metadata['bbox'][3]
 maxlon = metadata['bbox'][1]
 fsres = metadata['fsres']
 #eares = metadata['fsres_meters']

 #Retrieve mask metadata
 md = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % cdir)

 #cheight and maxlai
 vars = {
         'cheight':'/lustre/f2/dev/Nathaniel.Chaney/data/cheight/Simard_Pinto_3DGlobalVeg_JGR.tif',
         'maxlai':'/lustre/f2/dev/Nathaniel.Chaney/data/modis/mcd15a2h/stats/max.vrt'}
 tmp = {}
 for var in vars:
  file_in = vars[var]
  file_latlon = '%s/%s_latlon.tif' % (cdir,var)
  #os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon+360,minlat,maxlon+360,maxlat,fsres,fsres,file_in,file_latlon,log))
  os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_in,file_latlon,log))
  #file_ea = '%s/%s_ea.tif' % (cdir,var)
  #lproj = eaproj % float((maxlon+minlon)/2)
  #os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
  #tmp[var] = gdal_tools.read_raster(file_ea)
  tmp[var] = gdal_tools.read_raster(file_latlon)
  if (np.unique(tmp[var]).size == 1) & (np.unique(tmp[var])[0] == -9999):
   tmp[var] = 0.0
  else:
   tmp[var] = np.mean(tmp[var][tmp[var] != -9999])

 #1. Cutout the region of interest
 file_in = '/lustre/f2/dev/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img'
 #file_in = metadata['landcover']
 var = 'cover'
 file_latlon = '%s/%s_latlon.tif' % (cdir,var) 
 os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_in,file_latlon,log))
 #file_ea = '%s/%s_ea.tif' % (cdir,var)
 #lproj = eaproj % float((maxlon+minlon)/2)
 #os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
 #veg_file_ea = '%s/vegetation_ea.tif' % cdir
 #lu_file_ea = '%s/landuse_ea.tif' % cdir
 #cn_file_ea = '%s/cultivated2natural_ea.tif' % cdir
 #gt_file_ea = '%s/grass2tree_ea.tif' % cdir
 #de_file_ea = '%s/deciduous2evergreen_ea.tif' % cdir
 #ur_file_ea = '%s/natural2urban_ea.tif' % cdir
 #c3c4_file_ea = '%s/c32c4_ea.tif' % cdir
 veg_file_ll = '%s/vegetation_latlon.tif' % cdir
 lu_file_ll = '%s/landuse_latlon.tif' % cdir
 cn_file_ll = '%s/cultivated2natural_latlon.tif' % cdir
 gt_file_ll = '%s/grass2tree_latlon.tif' % cdir
 de_file_ll = '%s/deciduous2evergreen_latlon.tif' % cdir
 ur_file_ll = '%s/natural2urban_latlon.tif' % cdir
 c3c4_file_ll = '%s/c32c4_latlon.tif' % cdir

 #Construct the CLD lookup table
 table = {1:{'lu':2,'vg':0,'name':'Corn'},
          2:{'lu':2,'vg':1,'name':'Cotton'},
          3:{'lu':2,'vg':1,'name':'Rice'},
          4:{'lu':2,'vg':0,'name':'Sorghum'},
          5:{'lu':2,'vg':1,'name':'Soybeans'},
          6:{'lu':2,'vg':1,'name':'Sunflower'},
          10:{'lu':2,'vg':1,'name':'Peanuts'},
          11:{'lu':2,'vg':1,'name':'Tobacco'},
          12:{'lu':2,'vg':0,'name':'Sweet Corn'},
          13:{'lu':2,'vg':0,'name':'Pop or Orn Corn'},
          14:{'lu':2,'vg':1,'name':'Mint'},
          21:{'lu':2,'vg':1,'name':'Barley'},
          22:{'lu':2,'vg':1,'name':'Durum Wheat'},
          23:{'lu':2,'vg':1,'name':'Spring Wheat'},
          24:{'lu':2,'vg':1,'name':'Winter Wheat'},
          25:{'lu':2,'vg':1,'name':'Other Small Grains'},
          26:{'lu':2,'vg':1,'name':'Dbl Crop WinWht/Soybeans'},
          27:{'lu':2,'vg':1,'name':'Rye'},
          28:{'lu':2,'vg':1,'name':'Oats'},
          29:{'lu':2,'vg':0,'name':'Millet'},
          30:{'lu':2,'vg':1,'name':'Speltz'},
          31:{'lu':2,'vg':1,'name':'Canola'},
          32:{'lu':2,'vg':1,'name':'Flaxseed'},
          33:{'lu':2,'vg':1,'name':'Safflower'},
          34:{'lu':2,'vg':1,'name':'Rape Seed'},
          35:{'lu':2,'vg':1,'name':'Mustard'},
          36:{'lu':2,'vg':1,'name':'Alfalfa'},
          37:{'lu':2,'vg':1,'name':'Othey Hay/Non Alfalfa'},
          38:{'lu':2,'vg':1,'name':'Camelina'},
          39:{'lu':2,'vg':1,'name':'Buckwheat'},
          41:{'lu':2,'vg':1,'name':'Sugarbeets'},
          42:{'lu':2,'vg':1,'name':'Dry Beans'},
          43:{'lu':2,'vg':1,'name':'Potatoes'},
          44:{'lu':2,'vg':1,'name':'Other Crops'},
          45:{'lu':2,'vg':1,'name':'Sugarcane'},
          46:{'lu':2,'vg':1,'name':'Sweet Potatoes'},
          47:{'lu':2,'vg':1,'name':'Misc Vegs; Fruits'},
          48:{'lu':2,'vg':1,'name':'Watermelons'},
          49:{'lu':2,'vg':1,'name':'Onions'},
          50:{'lu':2,'vg':1,'name':'Cucumbers'},
          51:{'lu':2,'vg':1,'name':'Chick Peas'},
          52:{'lu':2,'vg':1,'name':'Lentils'},
          53:{'lu':2,'vg':1,'name':'Peas'},
          54:{'lu':2,'vg':1,'name':'Tomatoes'},
          55:{'lu':2,'vg':1,'name':'Caneberries'},
          56:{'lu':2,'vg':1,'name':'Hops'},
          57:{'lu':2,'vg':1,'name':'Herbs'},
          58:{'lu':2,'vg':1,'name':'Clover/Wildflowers'},
          59:{'lu':2,'vg':1,'name':'Sod/Grass Seed'},
          60:{'lu':2,'vg':0,'name':'Switchgrass'},
          61:{'lu':2,'vg':1,'name':'Fallow/Idle Cropland'},
          62:{'lu':1,'vg':1,'name':'Pasture/Grass'},
          63:{'lu':3,'vg':2,'name':'Forest'},
          64:{'lu':3,'vg':2,'name':'Shrubland'},
          65:{'lu':3,'vg':1,'name':'Barren'},
          66:{'lu':2,'vg':1,'name':'Cherries'},
          67:{'lu':2,'vg':1,'name':'Peaches'},
          68:{'lu':2,'vg':1,'name':'Apples'},
          69:{'lu':2,'vg':1,'name':'Grapes'},
          70:{'lu':2,'vg':1,'name':'Christmass Trees'},
          71:{'lu':2,'vg':1,'name':'Other Tree Crops'},
          72:{'lu':2,'vg':1,'name':'Citrus'},
          74:{'lu':2,'vg':1,'name':'Pecans'},
          75:{'lu':2,'vg':1,'name':'Almonds'},
          76:{'lu':2,'vg':1,'name':'Walnuts'},
          77:{'lu':2,'vg':1,'name':'Pears'},
          81:{'lu':3,'vg':1,'name':'Clouds/No Data'},
          82:{'lu':3,'vg':1,'name':'Developed'},
          83:{'lu':3,'vg':2,'name':'Water'},
          87:{'lu':3,'vg':2,'name':'Wetlands'},
          88:{'lu':3,'vg':2,'name':'Nonag/Undefined'},
          92:{'lu':3,'vg':2,'name':'Aquaculture'},
          111:{'lu':3,'vg':2,'name':'Open Water'},
          112:{'lu':3,'vg':1,'name':'Perennial Ice/Snow'},
          121:{'lu':3,'vg':1,'name':'Developed/Open Space'},
          122:{'lu':3,'vg':1,'name':'Developed/Low Intensity'},
          123:{'lu':3,'vg':1,'name':'Developed/Med Intensity'},
          124:{'lu':3,'vg':1,'name':'Developed/High Intensity'},
          131:{'lu':3,'vg':1,'name':'Barren'},
          141:{'lu':3,'vg':2,'name':'Deciduous Forest'},
          142:{'lu':3,'vg':4,'name':'Evergreen Forest'},
          143:{'lu':3,'vg':4,'name':'Mixed Forest'}, #CAREFUL
          152:{'lu':3,'vg':4,'name':'Shrubland'}, #CAREFUL
          171:{'lu':3,'vg':1,'name':'Grassland Herbaceous'},
          176:{'lu':3,'vg':1,'name':'Grassland/Pasture'},
          181:{'lu':1,'vg':1,'name':'Pasture/Hay'},
          190:{'lu':3,'vg':2,'name':'Woody Wetlands'},
          195:{'lu':3,'vg':2,'name':'Herbaceous Wetlands'},
          204:{'lu':2,'vg':1,'name':'Pistachios'},
          205:{'lu':2,'vg':1,'name':'Triticale'},
          206:{'lu':2,'vg':1,'name':'Carrots'},
          207:{'lu':2,'vg':1,'name':'Asparagus'},
          208:{'lu':2,'vg':1,'name':'Garlic'},
          209:{'lu':2,'vg':1,'name':'Cantaloupes'},
          210:{'lu':2,'vg':1,'name':'Prunes'},
          211:{'lu':2,'vg':1,'name':'Olives'},
          212:{'lu':2,'vg':1,'name':'Oranges'},
          214:{'lu':2,'vg':1,'name':'Broccoli'},
          216:{'lu':2,'vg':1,'name':'Peppers'},
          217:{'lu':2,'vg':1,'name':'Pomegranates'},
          218:{'lu':2,'vg':1,'name':'Nectarines'},
          219:{'lu':2,'vg':1,'name':'Greens'},
          220:{'lu':2,'vg':1,'name':'Plums'},
          221:{'lu':2,'vg':1,'name':'Strawberries'},
          222:{'lu':2,'vg':1,'name':'Squash'},
          223:{'lu':2,'vg':1,'name':'Apricots'},
          224:{'lu':2,'vg':1,'name':'Vetch'},
          225:{'lu':2,'vg':0,'name':'Dbl Crop WinWht/Corn'},
          226:{'lu':2,'vg':0,'name':'Dbl Crop Oats/Corn'},
          227:{'lu':2,'vg':1,'name':'Lettuce'},
          229:{'lu':2,'vg':1,'name':'Pumpkins'},
          230:{'lu':2,'vg':1,'name':'Dbl Crop Lettuce/Durum Wht'},
          231:{'lu':2,'vg':1,'name':'Dbl Crop Lettuce/Cantaloupe'},
          232:{'lu':2,'vg':1,'name':'Dbl Crop Lettuce/Cotton'},
          233:{'lu':2,'vg':1,'name':'Dbl Crop Lettuce/Barley'},
          234:{'lu':2,'vg':0,'name':'Dbl Crop Durum Wht/Sorghum'},
          235:{'lu':2,'vg':0,'name':'Dbl Crop Barley/Sorghum'},
          236:{'lu':2,'vg':0,'name':'Dbl Crop WinWht/Sorghum'},
          237:{'lu':2,'vg':0,'name':'Dbl Crop Barley/Corn'},
          238:{'lu':2,'vg':1,'name':'Dbl Crop WinWht/Cotton'},
          239:{'lu':2,'vg':1,'name':'Dbl Crop Soybeans/Cotton'},
          240:{'lu':2,'vg':1,'name':'Dbl Crop Soybeans/Oats'},
          241:{'lu':2,'vg':1,'name':'Dbl Crop Corn/Soybeans'},
          242:{'lu':2,'vg':1,'name':'Blueberries'},
          243:{'lu':2,'vg':1,'name':'Cabbage'},
          244:{'lu':2,'vg':1,'name':'Cauliflower'},
          245:{'lu':2,'vg':1,'name':'Celery'},
          246:{'lu':2,'vg':1,'name':'Rasishes'},
          247:{'lu':2,'vg':1,'name':'Turnips'},
          248:{'lu':2,'vg':1,'name':'Eggplants'},
          249:{'lu':2,'vg':1,'name':'Gourds'},
          250:{'lu':2,'vg':1,'name':'Cranberries'},
          254:{'lu':2,'vg':1,'name':'Dbl Crop Barley/Soybeans'},
         }

 cn_mapping = {1:0,2:0,3:1,4:1}
 gt_mapping = {0:0,1:0,2:1,3:1,4:1}
 de_mapping = {0:0,1:0,2:0,3:1,4:1}
 c3c4_mapping = {0:1,1:0,2:0,3:0,4:0}
 #2. Construct the vegetation type and landuse map
 cld_data = gdal_tools.read_raster(file_latlon)
 md = gdal_tools.retrieve_metadata(file_latlon)
 md['nodata'] = -9999.0
 ulcs = np.unique(cld_data)
 ulcs = ulcs[ulcs != 0]
 veg_data = np.copy(cld_data)
 lu_data = np.copy(cld_data)
 cn_data = np.copy(cld_data)
 gt_data = np.copy(cld_data)
 de_data = np.copy(cld_data)
 ur_data = np.copy(cld_data)
 c3c4_data = np.copy(cld_data)
 veg_data[:] = -9999.0
 lu_data[:] = -9999.0
 cn_data[:] = -9999.0
 gt_data[:] = -9999.0
 de_data[:] = -9999.0
 ur_data[:] = -9999.0
 c3c4_data[:] = -9999.0
 for ulc in ulcs:
  m = cld_data == ulc
  if ulc in table:
   veg_data[m] = table[ulc]['vg']
   lu_data[m] = table[ulc]['lu']
   cn_data[m] = cn_mapping[table[ulc]['lu']]
   gt_data[m] = gt_mapping[table[ulc]['vg']]   
   de_data[m] = de_mapping[table[ulc]['vg']]
   c3c4_data[m] = c3c4_mapping[table[ulc]['vg']]
  else:  
   veg_data[m] = 2 #temperature deciduous
   lu_data[m] = 3 #natural
   cn_data[m] = 1
   gt_data[m] = 1
   de_data[m] = 0
   c3c4_data[m] = 0
  #urban
  if ulc in [121,122,123,124]:ur_data[m] = 1
  else: ur_data[m] = 0
 #veg_data[veg_data == 0.0] = -9999.0 WHY?
 #lu_data[veg_data == 0.0] = -9999.0 WHY?
 gdal_tools.write_raster(lu_file_ll,md,lu_data)
 gdal_tools.write_raster(cn_file_ll,md,cn_data)
 gdal_tools.write_raster(gt_file_ll,md,gt_data)
 gdal_tools.write_raster(de_file_ll,md,de_data)
 gdal_tools.write_raster(ur_file_ll,md,ur_data)
 gdal_tools.write_raster(c3c4_file_ll,md,c3c4_data)

 #Retrieve the climate data
 pann = gdal_tools.read_raster('%s/pann_latlon.tif' % cdir)
 tann = gdal_tools.read_raster('%s/tann_latlon.tif' % cdir)
 t_cold = gdal_tools.read_raster('%s/t_cold_latlon.tif' % cdir)
 ncm = gdal_tools.read_raster('%s/ncm_latlon.tif' % cdir)

 #Correct for c3/c4 (LM code)
 m0 = (lu_data == 1) | (lu_data == 3) #Natural or pasture
 m = m0 & ((veg_data == 0) | (veg_data ==  1))
 m = m & (pann != -9999)
 # Rule based on analysis of ED global output; equations from JPC, 2/02
 temp = tann #Temperature from input climate data (deg K)
 precip = pann #Precip from input climate data (mm/yr)
 pc4=np.exp(-0.0421*(273.16+25.56-temp)-(0.000048*(273.16+25.5-temp)*precip));
 pt = np.ones(temp.shape) #Initialize to C3
 pt[pc4 > 0.5] = 0 #Set to C4
 veg_data[m] = pt[m]

 #output the species file
 gdal_tools.write_raster(veg_file_ll,md,veg_data)

 return

def create_soil_products(metadata,cdir,workspace,log):

 if metadata['soil']['type'] == 'polaris':
  create_soil_products_polaris(metadata,cdir,workspace,log)
 elif metadata['soil']['type'] == 'soilgrids':
  create_soil_products_soilgrids(metadata,cdir,workspace,log)

 return

def create_soil_products_soilgrids(metadata,cdir,workspace,log):

 from rpy2.robjects import r,FloatVector
 r('library("soiltexture")')

 #Define metadata
 minlat = metadata['bbox'][2]
 minlon = metadata['bbox'][0]
 maxlat = metadata['bbox'][3]
 maxlon = metadata['bbox'][1]
 fsres = metadata['fsres']
 eares = metadata['fsres_meters']
 ddir = metadata['soil']['dir']

 #Retrieve mask metadata
 #md = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % cdir)
 md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)

 #Extract properties for region (Top layer for now)
 vars = ['ORCDRC','BLDFIE','SLTPPT','CLYPPT','SNDPPT']
 #layers = {1:0,2:5,3:15,4:30,5:60,6:100,7:200}
 layers = {1:0,7:200}
 for layer in layers:
  data = {}
  for var in vars:
   print(var,layer)
   #file_in = '%s/%s' % (metadata['soil_dataset'],vars[var])
   #file_in = vars[var]#'%s/%s' % (metadata['soil_dataset'],vars[var])
   file_in = '%s/%s_M_sl%d_250m_ll.tif' % (ddir,var,layer)
   #print file_in
   file_latlon = '%s/%s_%dcm_latlon.tif' % (cdir,var,layers[layer])
   file_ea = '%s/%s_%dcm_ea.tif' % (cdir,var,layers[layer])
   #Cut out the region
   os.system('gdalwarp -ot Float32 -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f %s %s >& %s' % (fsres,fsres,minlon,minlat,maxlon,maxlat,file_in,file_latlon,log))
   #Reproject region to equal area
   lproj = eaproj % float((maxlon+minlon)/2)
   os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
   # EZDEV: note, I did not change -r average to -r lanczos here - soil!
   #Read in the data
   #data[var] = gdal_tools.read_raster(file_latlon).astype(np.float32)
   data[var] = gdal_tools.read_raster(file_ea).astype(np.float32)
   #data[var] = np.ma.masked_array(data[var],data[var]==-9999)

  #Extract sand, silt, clay, and organic matter
  S = data['SNDPPT']
  C = data['CLYPPT']
  ST = data['SLTPPT']
  OM = data['ORCDRC']
  OM[OM != -9999] = 100*OM[OM != -9999]/1000.0 #%
 
  #Make sure that S,C,and ST add up too 100
  m = (S != -9999) & (C != -9999) & (ST != -9999)
  T = S[m] + C[m] + ST[m]
  S[m] = 100/T*S[m]
  C[m] = 100/T*C[m]
  ST[m] = 100/T*ST[m]
  S[(np.isnan(S) == 1) | (np.isinf(S) == 1)] = -9999
  C[(np.isnan(C) == 1) | (np.isinf(C) == 1)] = -9999
  ST[(np.isnan(ST) == 1) | (np.isinf(ST) == 1)] = -9999

  #Mask out
  #S = np.ma.masked_array(S,S==-9999)
  #C = np.ma.masked_array(C,C==-9999)
  #ST = np.ma.masked_array(ST,ST==-9999)
  #OM = np.ma.masked_array(OM,OM==-9999)

  #QC
  OM[OM > 8] = 8
  C[C > 60] = 60
  S[(S < 5) & (S != -9999)] = 5
  C[(C < 3) & (3 != -9999)] = 3
  #OM[(OM < 0.01) & (OM != -9999)] = 0.01
  #C[(C < 0) & (C != -9999)] = 0
  #C[(S < 0) & (S != -9999)] = 0

  #Convert to the right units
  mask = (S != -9999) & (C != -9999) & (OM != -9999)
  S[mask] = S[mask]/100
  C[mask] = C[mask]/100
  ST[mask] = ST[mask]/100

  #Compute thetas,theta33,theta1500,and ksat
  thetas = np.zeros(S.shape).astype(np.float32)
  theta33 = np.zeros(S.shape).astype(np.float32)
  theta1500 = np.zeros(S.shape).astype(np.float32)
  ksat = np.zeros(S.shape).astype(np.float32)
  thetas[:] = -9999
  theta33[:] = -9999
  theta1500[:] = -9999
  ksat[:] = -9999
  thetas[mask] = pedotransfer.ThetaS_Saxton2006(S[mask],C[mask],OM[mask])
  theta33[mask] = pedotransfer.Theta_33_Saxton2006(S[mask],C[mask],OM[mask])
  theta1500[mask] = pedotransfer.Theta_1500_Saxton2006(S[mask],C[mask],OM[mask])
  ksat[mask] = pedotransfer.Ksat_Saxton2006(S[mask],C[mask],OM[mask])/10 #cm/hr
  #thetas = pedotransfer.ThetaS_Saxton2006(S,C,OM)
  #theta33 = pedotransfer.Theta_33_Saxton2006(S,C,OM)
  #theta1500 = pedotransfer.Theta_1500_Saxton2006(S,C,OM)
  #ksat = pedotransfer.Ksat_Saxton2006(S,C,OM)

  #Deal with completely empty case
  if ((len(np.unique(np.ma.getdata(ksat))) == 1) & (np.unique(np.ma.getdata(ksat))[0] == -9999)):
   S[:] = 60.0
   C[:] = 10.0
   ST[:] = 30.0
   OM[:] = 0.0
   thetas[:] = 0.4
   ksat[:] = 0.003
   theta33[:] = 0.2
   theta1500[:] = 0.1

  #Calculate the corresponding FAO texture class
  #tclass = pedotransfer.FAO_Soil_Texture(np.ma.masked_array(S,S==-9999),
  #              np.ma.masked_array(C,C==-9999),np.ma.masked_array(ST,ST==-9999))
  #tclass = np.ma.getdata(tclass)
  tclass = np.copy(C)
  tclass[:] = -9999
  #Calculate the corresponding soil texture
  clay = C[m]
  sand = S[m]
  silt = ST[m]
  tmp = clay+sand+silt
  clay = clay*100/tmp
  sand = sand*100/tmp
  silt = silt*100/tmp
  r.assign('clay',FloatVector(clay))
  r.assign('sand',FloatVector(sand))
  r.assign('silt',FloatVector(silt))
  r.assign('om',FloatVector(OM[m]))
  #Compute the texture class
  r('my.text <- data.frame("CLAY" = clay,"SILT" = silt,"SAND" = sand,"OC" = om)') #
  #Classify according to the USDA classification
  r('output <- TT.points.in.classes(tri.data = my.text,class.sys = "USDA.TT")')
  #Import back into python
  output = np.array(r("output"))
  tmp = np.sum(output,axis=1)
  mask = np.where(tmp > 1)[0]
  for i in mask:
   tmp = np.where(output[i,:] > 1)[0]
   tmp2 = np.zeros(output[i,:].size)
   tmp2[tmp[0]] = 1
   output[i,:] = tmp2
  tmp = np.where(output > 0)[1] + 1
  #Identify points with clay above 60% and set to heavy clay
  tmp[clay > 60] = 0
  #Map to class
  mapping = np.array([0,2,1,7,4,3,9,8,6,10,5,11,12])
  #Original: HEC Cl SiCl SaCl ClLo SiClLo SaClLo Lo SiLo SaLo Si LoSa Sa
  #Original: hec lic sic sac cl sicl sacl l sil sal si ls s
  #'hec'->heavy clay (0)
  #'sic'->silty clay (1)
  #'lic'->(light) clay (2)
  #'sicl'->silty clay loam (3)
  #'cl'->clay loam (4)
  #'si'->silt (5)
  #'sil'->silt loam (6)
  #'sac'->sandy clay (7)
  #'l'->loam (8)
  #'sacl'->sandy clay loam (9)
  #'sal'->sandy loam (10)
  #'ls'->loamy sand (11)
  #'s'->sand (12)
  soil_texture  = mapping[tmp]
  tclass[m] = soil_texture

  #Save info
  output = {'soiltexture':tclass,'ThetaS':thetas,'Theta33':theta33,'Theta1500':theta1500,'Ksat':ksat}
  mask = (S != -9999) & (C != -9999) & (OM != -9999)

  #Clean up the variables
  for var in output:
   output[var][mask == 0] = -9999.0

  #Output file
  md['nodata'] = -9999.0
  for var in output:
   file = '%s/%s_%dcm_ea.tif' % (cdir,var,layers[layer])
   gdal_tools.write_raster(file,md,output[var])

  #Output file
  #md['nodata'] = -9999.0
  #for var in output:
  # file = '%s/%s_latlon.tif' % (cdir,var)
  # gdal_tools.write_raster(file,md,output[var])

 #Exit R
 del r,FloatVector

 return

def create_soil_products_polaris(metadata,cdir,workspace,log):

 #Define metadata
 minlat = metadata['bbox'][2]
 minlon = metadata['bbox'][0]
 maxlat = metadata['bbox'][3]
 maxlon = metadata['bbox'][1]
 fsres = metadata['fsres']

 #Retrieve mask metadata
 md = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % cdir)

 #Extract properties for region (Top layer for now)
 vars = {'BDTICM':'/lustre/f2/dev/Nathaniel.Chaney/data/soilgrids/BDTICM_M_250m_ll.tif',
         'ORCDRC':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/om_mean_0_5.vrt',#'ORCDRC_M_sl1_250m_ll.tif',
         'BLDFIE':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/bd_mean_0_5.vrt',#'BLDFIE_M_sl1_250m_ll.tif',
         'SLTPPT':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/silt_mean_0_5.vrt',#'SLTPPT_M_sl1_250m_ll.tif',
         'CLYPPT':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/clay_mean_0_5.vrt',#'CLYPPT_M_sl1_250m_ll.tif',
         'SNDPPT':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/sand_mean_0_5.vrt',#'SNDPPT_M_sl1_250m_ll.tif'}
         #'ksat':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/ksat_mean_0_5.vrt',
         #'theta_s':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/theta_s_mean_0_5.vrt',
         #'hb':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/hb_mean_0_5.vrt',
         #'lambda':'/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt/lambda_mean_0_5.vrt',
         }
 data = {}
 for var in vars:
  #file_in = '%s/%s' % (metadata['soil_dataset'],vars[var])
  file_in = vars[var]#'%s/%s' % (metadata['soil_dataset'],vars[var])
  file_latlon = '%s/%s_latlon.tif' % (cdir,var)
  #Cut out the region
  os.system('gdalwarp -ot Float32 -dstnodata -9999 -te %.16f %.16f %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,file_in,file_latlon,log))
  #Read in the data
  data[var] = gdal_tools.read_raster(file_latlon)
  data[var] = np.ma.masked_array(data[var],data[var]==-9999)

 #Redefine ksat
 vrt_dir = '/lustre/f2/dev/Nathaniel.Chaney/data/POLARIS-LSM/50mpc/tiff/vrt'
 data['ksat'] = soil_properties.compute_cwa_median('ksat',vrt_dir,cdir,minlon,minlat,maxlon,maxlat,log)
 data['theta_s'] = soil_properties.compute_cwa_median('theta_s',vrt_dir,cdir,minlon,minlat,maxlon,maxlat,log)
 data['theta_33'] = soil_properties.compute_cwa_median('theta_33',vrt_dir,cdir,minlon,minlat,maxlon,maxlat,log)
 data['theta_1500'] = soil_properties.compute_cwa_median('theta_1500',vrt_dir,cdir,minlon,minlat,maxlon,maxlat,log)

 S = data['SNDPPT'] #%
 C = data['CLYPPT'] #%
 ST = data['SLTPPT'] #%
 #OM = 100*data['ORCDRC']/1000.0 #% (soil grids)
 OM = data['ORCDRC'] #% (POLARIS)
 thetas = data['theta_s']
 theta33 = data['theta_33']
 theta1500 = data['theta_1500']
 ksat = data['ksat']#cm/hr ##*10.0/3600.0 #mm/s
 #ksat[ksat != -9999] = ksat[ksat != -9999]*10.0/3600.0 #mm/s
 #psisat = -data['hb']/100 #meters
 #psisat[psisat != -9999] = -psisat[psisat != -9999]/100 #meters
 #chb = 1/data['lambda']
 #chb[chb != -9999] = 1/chb[chb != -9999]
 
 #dat_w_sat=[0.380, 0.445, 0.448, 0.412, 0.414, 0.446, 0.424, 0.445, 0.445, 0.0, 0.0, 0.0, 0.0, 0.0]
 #dat_k_sat_ref=[0.021, .0036, .0018, .0087, .0061, .0026, .0051, .0036, .0036, 0.0, 0.0, 0.0, 0.0, 0.0]
 #dat_psi_sat_ref=[-.059, -0.28, -0.27, -0.13, -0.13, -0.27, -0.16, -0.28, -0.28, 0.0, 0.0, 0.0, 0.0, 0.0]
 #dat_chb=[3.5,   6.4,  11.0,   4.8,   6.3,   8.4,   6.3,   6.4,   6.4, 0.0, 0.0, 0.0, 0.0, 0.0]

 #Deal with completely empty case
 if ((len(np.unique(np.ma.getdata(S))) == 1) & (np.unique(np.ma.getdata(S))[0] == -9999)):
  S[:] = 60.0
  C[:] = 10.0
  ST[:] = 30.0
  OM[:] = 0.0
  thetas[:] = 0.4
  ksat[:] = 0.003
  #psisat[:] = -.059
  #chb[:] = 3.5
  theta33[:] = 0.2
  theta1500[:] = 0.1
 #ksat[ksat == -9999] = 0.021
 #thetas[thetas == -9999] = 0.4
 #psisat[psisat == -9999] = -0.059
 #chb[chb == -9999] = 3.5
 

 #Make sure that S,C,and ST add up too 100
 T = S + C + ST
 S = 100/T*S
 C = 100/T*C
 ST = 100/T*ST
 S[(np.isnan(S) == 1) | (np.isinf(S) == 1)] = -9999
 C[(np.isnan(C) == 1) | (np.isinf(C) == 1)] = -9999
 ST[(np.isnan(ST) == 1) | (np.isinf(ST) == 1)] = -9999

 #Calculate the corresponding FAO texture class
 #soil_class = pedotransfer.FAO_Soil_Texture(S,C,ST)
 tclass = pedotransfer.FAO_Soil_Texture(S,C,ST)

 #Use pedotransfer functions to estimate all the necessary properties
 #output = {'FAOtexture':tclass,'ThetaS':thetas,'Ksat':ksat,'PsiSat':psisat,'Chb':chb}
 output = {'FAOtexture':tclass,'ThetaS':thetas,'Ksat':ksat,'Theta33':theta33,'Theta1500':theta1500}
 mask = (S != -9999) & (C != -9999) & (OM != -9999)
 #print np.sum(mask),mask.shape

 #Clean up the variables
 for var in output:
  output[var][mask == 0] = -9999.0

 #Output file
 md['nodata'] = -9999.0
 for var in output:
  file = '%s/%s_latlon.tif' % (cdir,var)
  gdal_tools.write_raster(file,md,output[var])

 return

def determine_component_fractions(cdir,metadata):
 
 minlat = metadata['bbox'][2]
 minlon = metadata['bbox'][0]
 maxlat = metadata['bbox'][3]
 maxlon = metadata['bbox'][1]
 lat = (minlat+maxlat)/2
 lon = (minlon+maxlon)/2

 #Read in the glacier mask
 #glaciers_ea = gdal_tools.read_raster('%s/glaciers_latlon.tif' % (cdir,))
 glaciers_ea = gdal_tools.read_raster('%s/glaciers_ea.tif' % (cdir,))

 #Read in the lake mask
 #lakes_ea = gdal_tools.read_raster('%s/lakes_latlon.tif' % (cdir,))
 lakes_ea = gdal_tools.read_raster('%s/lakes_ea.tif' % (cdir,))
 
 #Read in the actual mask
 #mask_ea = gdal_tools.read_raster('%s/mask_latlon.tif' % (cdir,))
 mask_ea = gdal_tools.read_raster('%s/mask_ea.tif' % (cdir,))

 #Set the soils mask to the actual land mask
 soils_ea = np.copy(mask_ea)

 #Priority Glaciers>Lakes>Soil
 glaciers_ea[mask_ea != 1] = -9999
 lakes_ea[(mask_ea != 1) | (glaciers_ea != -9999)] = -9999
 soils_ea[(mask_ea != 1) | (lakes_ea != -9999) | (glaciers_ea != -9999)] = -9999

 #Output the masks
 #md = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % (cdir,))
 #md['nodata'] = -9999
 #gdal_tools.write_raster('%s/glaciers_latlon.tif' % (cdir,),md,glaciers_ea)
 #gdal_tools.write_raster('%s/lakes_latlon.tif' % (cdir,),md,lakes_ea)
 #gdal_tools.write_raster('%s/soils_latlon.tif' % (cdir,),md,soils_ea)
 md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % (cdir,))
 md['nodata'] = -9999
 gdal_tools.write_raster('%s/glaciers_ea.tif' % (cdir,),md,glaciers_ea)
 gdal_tools.write_raster('%s/lakes_ea.tif' % (cdir,),md,lakes_ea)
 gdal_tools.write_raster('%s/soils_ea.tif' % (cdir,),md,soils_ea)

 #Compute the fractions
 soil_frac = float(np.sum(soils_ea != -9999))/float(np.sum(mask_ea == 1))
 lake_frac = float(np.sum(lakes_ea != -9999))/float(np.sum(mask_ea == 1))
 glacier_frac = float(np.sum(glaciers_ea != -9999))/float(np.sum(mask_ea == 1))

 #Set some restrictions
 lat = (minlat+maxlat)/2
 if (lat < -60):
  lake_frac = 0.0
  soil_frac = 0.0
  glacier_frac = 1.0
 #if (lake_frac < 0.001):
 # soil_frac = soil_frac/(1-lake_frac)
 # glacier_frac = glacier_frac/(1-lake_frac)
 # lake_frac = 0.0
 #Override fractions if desired
 if metadata['land_fractions'] == 'original':
  #lake fraction
  #fp = nc.Dataset('%s/river/river_data.nc' % metadata['dir'])
  #lons = fp.variables['grid_x'][:]
  #lats = fp.variables['grid_y'][:]
  #Change to -180 to 180 if necessary
  #lons[lons > 180] = lons[lons > 180] - 360
  #Find the match
  #ilat = np.argmin(np.abs(lats-lat))
  #ilon = np.argmin(np.abs(lons-lon))
  tile = metadata['tile']
  ilat = metadata['ilat']
  ilon = metadata['ilon']
  file = metadata['rn_template']
  file = file.replace('$tid',str(tile))
  fp = nc.Dataset(file)
  tmp = np.ma.getdata(fp.variables['lake_frac'][ilat,ilon])
  if lake_frac == 0:tmp = 0
  fp.close()
  if np.sum(glacier_frac) > 1:
    glacier_frac = glacier_frac/np.sum(glacier_frac)
  if (np.sum(glacier_frac) + np.sum(lake_frac)) > 1:
    lake_frac = lake_frac*(1-np.sum(glacier_frac))/np.sum(lake_frac)
  if soil_frac > 0:
     soil_frac = soil_frac*(1-np.sum(lake_frac)-np.sum(glacier_frac))/np.sum(soil_frac)

 return (soil_frac,glacier_frac,lake_frac)

def create_meteorology_products(metadata,cdir,workspace,log):

 if metadata['climate']['type'] == 'worldclim':
  create_meteorology_products_worldclim(metadata,cdir,workspace,log)
 elif metadata['climate']['type'] == 'prism':
  create_meteorology_products_prism(metadata,cdir,workspace,log)
 else:
  print("Unknown climate database -> Cannot process")
  exit()
 
 return 

def create_meteorology_products_prism(metadata,cdir,workspace,log):

 return

def create_meteorology_products_worldclim(metadata,cdir,workspace,log):

 #Define metadata
 minlat = metadata['bbox'][2]
 minlon = metadata['bbox'][0]
 maxlat = metadata['bbox'][3]
 maxlon = metadata['bbox'][1]
 fsres = metadata['fsres']
 eares = metadata['fsres_meters']

 #Retrieve mask metadata
 #md = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % cdir)
 md = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)

 #1. Cutout the region of interest for the 12 months
 ddir = metadata['climate']['dir']#/lustre/f2/dev/Nathaniel.Chaney/data/worldclim'
 wdir = '%s/workspace' % cdir
 data = {}
 for var in ['tavg','prec','wind','vapr','srad']:
  for month in range(0,12):
   file_in = '%s/wc2.0_30s_%s_%02d.tif' % (ddir,var,month+1)
   file_latlon = '%s/%s_%02d_latlon.tif' % (wdir,var,month+1) 
   os.system('gdalwarp -t_srs EPSG:4326 -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,fsres,fsres,file_in,file_latlon,log))
   file_ea = '%s/%s_%02d_ea.tif' % (wdir,var,month+1)
   lproj = eaproj % float((maxlon+minlon)/2)
   os.system('gdalwarp -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
   #Read it in
   if var not in data:data[var] = []
   if var in ['tavg','prec']:
    data[var].append(gdal_tools.read_raster(file_ea).astype(np.float32))
    #data[var].append(gdal_tools.read_raster(file_latlon).astype(np.float32))
   #Retrieve metadata
   md = gdal_tools.retrieve_metadata(file_ea)
   #md = gdal_tools.retrieve_metadata(file_latlon)

 #Convert to arrays
 for var in data:
  data[var] = np.array(data[var])
 data['tavg'] = data['tavg'] + 273.15 #K
   
 #Annual precipitation
 pann = np.sum(data['prec'],axis=0)
 #Annual temperature
 tann = np.mean(data['tavg'],axis=0)
 #Average temperature of the coldest month
 t_cold = np.min(data['tavg'],axis=0)
 #Number of cold months
 m = data['tavg'] < 283.0
 ncm = np.sum(m,axis=0)
 #Create mask
 m = (pann < 0) | (tann < 0)
 if np.sum(~m) > 0:
  pann[m] = np.mean(pann[~m])#-9999
  tann[m] = np.mean(tann[~m])#-9999
  t_cold[m] = np.mean(t_cold[~m])#-9999
  ncm[m] = np.mean(ncm[~m])#-9999
 else:
  pann[m] = 0.0
  tann[m] = 283.16
  t_cold[m] = 283.16
  ncm[m] = 0

 #Output files
 md['nodata'] = -9999
 gdal_tools.write_raster('%s/pann_ea.tif' % cdir,md,pann)
 gdal_tools.write_raster('%s/tann_ea.tif' % cdir,md,tann)
 gdal_tools.write_raster('%s/t_cold_ea.tif' % cdir,md,t_cold)
 gdal_tools.write_raster('%s/ncm_ea.tif' % cdir,md,ncm)
 #gdal_tools.write_raster('%s/pann_latlon.tif' % cdir,md,pann)
 #gdal_tools.write_raster('%s/tann_latlon.tif' % cdir,md,tann)
 #gdal_tools.write_raster('%s/t_cold_latlon.tif' % cdir,md,t_cold)
 #gdal_tools.write_raster('%s/ncm_latlon.tif' % cdir,md,ncm)

 return

def Create_Unified_Tile_Map_EA(tiles,tid,ttype,cdir):
    # EZDEV added
    #Retrieve metadata
    print('tile types = {}'.format(ttype))
    print('tile ids = {}'.format(tid))
    undef = -9999.0
    metadata = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % (cdir,))
    metadata['nodata'] = undef  
    #Read in the soil tiles
    if 3 in ttype:
     sdata = gdal_tools.read_raster('%s/soil_tiles_ea.tif' % (cdir,))
     shape = sdata.shape

    #Read in the lake tiles
    if 2 in ttype: 
     ldata = gdal_tools.read_raster('%s/lakes_ea.tif' % (cdir,))
     ldata[ldata != undef] = 1
     shape = ldata.shape

    #Read in the glacier tiles
    if 1 in ttype:
     gdata = gdal_tools.read_raster('%s/glaciers_ea.tif' % (cdir,))
     gdata[gdata != undef] = 1
     shape = gdata.shape    

    #Create the unified tile map
    udata = np.zeros(shape)
    udata[:] = undef
    for it in range(len(tiles)):
     if ttype[it] == 1:
      mask = gdata == tid[it]
     elif ttype[it] == 2:
      mask = ldata == tid[it]
     elif ttype[it] == 3:
      mask = sdata == tid[it]
     else:
      print("tiles and tile types::")
      print(tiles)
      print(ttype)
      raise Exception("unknown tile type found!")

     udata[mask] = tiles[it]    
    #Write out the map
    gdal_tools.write_raster('%s/tiles_ea.tif' % cdir,metadata,udata)   
    return

def Create_Unified_Tile_Map(tiles,tid,ttype,cdir):

 #Retrieve metadata
 undef = -9999.0
 metadata = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % (cdir,))
 metadata['nodata'] = undef

 #Read in the soil tiles
 if 3 in ttype:
  sdata = gdal_tools.read_raster('%s/soil_tiles_latlon.tif' % (cdir,))
  shape = sdata.shape
 
 #Read in the lake tiles
 if 2 in ttype: 
  ldata = gdal_tools.read_raster('%s/lake_tiles_latlon.tif' % (cdir,))
  shape = ldata.shape
 
 #Read in the glacier tiles
 if 1 in ttype:
  gdata = gdal_tools.read_raster('%s/glacier_tiles_latlon.tif' % (cdir,))
  shape = gdata.shape

 #Create the unified tile map
 udata = np.zeros(shape)
 udata[:] = undef
 for it in range(len(tiles)):
  if ttype[it] == 1:
   mask = gdata == tid[it]
  elif ttype[it] == 2:
   mask = ldata == tid[it]
  elif ttype[it] == 3:
   mask = sdata == tid[it]
  udata[mask] = tiles[it]

 #Write out the map
 gdal_tools.write_raster('%s/tiles.tif' % cdir,metadata,udata)

 return


def create_grid_cell_database(id, metadata, tile, y, x):
   
    #Retrieve info
    buffer = metadata['hillslope']['land_buffer'] # EZDEV
    metadata['tile'] = tile
    metadata['ilat'] = y - 1
    metadata['ilon'] = x - 1
    minlat = metadata['bbox'][2]
    minlon = metadata['bbox'][0]
    maxlat = metadata['bbox'][3]
    maxlon = metadata['bbox'][1]
    lat = (minlat+maxlat)/2
    lon = (minlon+maxlon)/2
    fsres = metadata['fsres']
    ldir = metadata['ldir']
    file_shp = '%s/shapefile/grid.shp' % metadata['dir']
   
    #General parameters
    ntile = 0
    information = {}
   
    #Create the cell directory
    cid = 'tile:%d,is:%d,js:%d' % (tile,y,x)
    cdir = '%s/tile:%d,is:%d,js:%d' % (ldir,tile,y,x)
    os.system('rm -rf %s' % cdir) #HERE
    os.system('mkdir -p %s' % cdir)
   
    #Create workspace
    workspace = '%s/workspace' % cdir
    # REMOVE TO AVOID WIPING WORKSPACE # EZDEV
    os.system('rm -rf %s' % workspace) #HERE
    os.system('mkdir -p %s' % workspace)
   
    #Define log
    log = '%s/log.txt' % cdir
   
    #Determine the node id
    nid = determine_nid()
   
   ############################################
    #Print cid to node id log (beginning)
    os.system('echo begin:%s >> %s/workspace/nid/%s.txt' % (cid,metadata['dir'],nid))
    #Create the mask
    print(nid,cid,'Preparing cell mask', memory_usage())
    #create_mask(metadata,id,cdir,log)
    rp(create_mask,(metadata,id,cdir,log))
   
    #Create the buffer mask
    print(nid,cid,'Preparing cell buffered mask',memory_usage())
    #create_mask_buffered(metadata,id,cdir,log,buffer)
    rp(create_mask_buffered,(metadata,id,cdir,log,buffer))
   
    #Prepare meteorology products
    print(nid,cid,'Preparing meteorology products',memory_usage())
    #create_meteorology_products(metadata,cdir,workspace,log)
    rp(create_meteorology_products,(metadata,cdir,workspace,log))
   
    #Prepare the land cover products
    print(nid,cid,'Preparing land cover products',memory_usage())
    #create_landcover_products(metadata,cdir,workspace,log)
    rp(create_landcover_products,(metadata,cdir,workspace,log))
   
    #Prepare the water management products
    print(nid,cid,'Preparing water management products',memory_usage())
    #create_watermanagement_products(metadata,cdir,workspace,log)
    rp(create_watermanagement_products,(metadata,cdir,workspace,log))
   
    #Prepare the geohydrology products
    print(nid,cid,'Preparing geohydrology products',memory_usage())
    #create_geohydrology_products(metadata,cdir,workspace,log)
    rp(create_geohydrology_products,(metadata,cdir,workspace,log))
   
    #Prepare the soil products
    print(nid,cid,'Preparing soil products',memory_usage())
    #create_soil_products(metadata,cdir,workspace,log)
    rp(create_soil_products,(metadata,cdir,workspace,log))
   
    #Prepare the terrain products
    print(nid,cid,'Preparing terrain products',memory_usage())
    #create_terrain_products(metadata,id,cdir,workspace,log,nid,cid)
    rp(create_terrain_products,(metadata,id,cdir,workspace,log,nid,cid))
   
    #Prepare glacier map
    print(nid,cid,'Preparing glacier products',memory_usage())
    create_glacier_products(metadata,cdir,workspace,log)
   
    #Prepare lake products
    print(nid,cid,'Preparing lake products',memory_usage())
    create_lake_products(metadata,cdir,workspace,log)
   #############################################
   
   
    #Determine the soil, glacier, and lake masks and define the fractions
    print(nid,cid,'Determining the fractions',memory_usage())
    (soil_frac,glacier_frac,lake_frac) = determine_component_fractions(cdir,metadata)
   
    #Exit if the sum is now 0
    if (soil_frac + glacier_frac + lake_frac) == 0:
        #Remove directory
        os.system('rm -rf %s' % cdir)
        print(nid,cid,"Fractions add to 0. Existing for cell %s" % cdir.split('/')[-1])
        return
   
    #Retrieve lake information
    if lake_frac > 0:
        print(nid,cid,'Retrieving the lake information',memory_usage())
        tile = metadata['tile']
        ilat = metadata['ilat']
        ilon = metadata['ilon']
        file = metadata['rn_template']
        lake_file = file.replace('$tid',str(tile))
        lake_information = lake_properties.Extract_Lake_Properties(cdir,lake_file,lake_frac,metadata)
        #print 'lake',lake_information
      
        #Update ntile
        ntile += lake_information['ntile']
      
        #Add to the general dictionary
        information['lake'] = lake_information
   
    #Retrieve glacier information
    if glacier_frac > 0:
        print(nid,cid,'Retrieving the glacier information',memory_usage())
        glacier_information = glacier_properties.Extract_Glacier_Properties(cdir,glacier_frac)
        #print 'glacier',glacier_information
      
        #Update ntile
        ntile += glacier_information['ntile']
      
        #Add to the general dictionary
        information['glacier'] = glacier_information
   
    #If overriding then set the new lake,glacier,and soil fractions
    #f metadata['frac_override'] == True:
    #if lake_frac > 0:lake_frac = lake_information['frac']
    #if (np.sum(glacier_frac) + np.sum(lake_frac)) > 1:
    #  lake_frac = lake_frac*(1-np.sum(glacier_frac))/np.sum(lake_frac)
    #soil_frac = soil_frac*(1-np.sum(lake_frac)-np.sum(glacier_frac))/np.sum(soil_frac)
    #if lake_frac > 0:lake_information['frac'] = lake_frac
   
    #Retrieve hillslope information
    if soil_frac > 0:
        print(nid,cid,'Retrieving the hillslope information',memory_usage())
        #hillslope_information = hillslope_properties.Extract_Hillslope_Properties(cdir,metadata,soil_frac,buffer)
        if metadata['hillslope']['type'] == 'original':
            hillslope_information = hillslope_properties.Extract_Hillslope_Properties_Original(metadata,soil_frac)
        else:
            hillslope_information = hillslope_properties.Extract_Hillslope_Properties_Updated(cdir,metadata,soil_frac,buffer,log,cid)
        # print('hillslope',hillslope_information) # EZDEV
      
        #Retrieve soil properties
        print(nid,cid,'Retrieving the soil information',memory_usage())
        if metadata['soil']['type'] == 'original':
            soil_information = soil_properties.Extract_Soil_Properties_Original(lat,lon,metadata)
        else:
            soil_information = soil_properties.Extract_Soil_Properties(cdir,metadata)
      
        #Retrieve geohydrology properties
        print(nid,cid,'Retrieving the geohydrology information',memory_usage())
        geohydrology_information = geohydrology_properties.Extract_Geohydrology_Properties(cdir,metadata)
        #print 'geohydrology',geohydrology_information
       
        #Retrieve vegetation properties
        print(nid,cid,'Retrieving the vegetation information',memory_usage())
        vegetation_information = vegetation_properties.Extract_Vegetation_Properties(cdir,metadata)
        #print 'vegetation',vegetation_information
      
        #Put all info in the soil dictionary
        for var in vegetation_information:   
            soil_information[var] = vegetation_information[var]
        for var in geohydrology_information:
            soil_information[var] = geohydrology_information[var]
        for var in hillslope_information:
            soil_information[var] = hillslope_information[var]
      
        #Update ntile
        ntile += hillslope_information['ntile']
      
        #Add to the general dictionary
        information['soil'] = soil_information
      
    #Create the group for the grid cell
    #print 'Placing the data in the hdf5 virtual file'
    print(nid,cid,'Placing the data in the output file',memory_usage())
    #Create the output file
    file = '%s/land_model_input_database.h5' % (cdir,)
    grp = h5py.File(file,'w',driver='core',backing_store=True)
    
    #Create the new dataset
    grp['nband'] = np.int32(2)
    grp['ntile'] = np.int32(ntile)
    mdgrp = grp.create_group('metadata')
    mdgrp['latitude'] = np.array([lat,])
    mdgrp['longitude'] = np.array([lon,])
   
    #Place the general parameters
    misc = {'glacier':{'ttype':1},'lake':{'ttype':2},'soil':{'ttype':3}}
    frac,ttype,tid = [],[],[]
    for var in ['glacier','lake','soil']:
        if var in information:   
            #fractional coverage
            frac = frac + list(information[var]['frac']) 
            #type of cell
            tmp = misc[var]['ttype']*np.ones(len(information[var]['frac']))
            ttype = ttype + list(tmp)
            #tile id (independent)
            tid = tid + list(np.arange(information[var]['ntile'])+1)
    tiles = np.arange(ntile)
    frac = np.array(frac)
    ttype = np.array(ttype)
    tid = np.array(tid)
    mdgrp['frac'] = np.float64(frac)
    mdgrp['tile'] = np.int32(tiles[:])
    mdgrp['type'] = np.int32(ttype[:])
    mdgrp['tid'] = np.int32(tid[:])
   
    #Create a unified tile map
    if metadata['hillslope']['type'] != 'original':
        print(nid,cid,"Creating a unified tile map",memory_usage())
        Create_Unified_Tile_Map(tiles,tid,ttype,cdir)

        # EZDEV ADDED::
        print(nid,cid,"Creating a unified tile map, ea",memory_usage())
        Create_Unified_Tile_Map_EA(tiles,tid,ttype,cdir)
   
    #Retrieve meteorology properties
    print(nid,cid,'Retrieving the meteorology information',memory_usage())
    meteorology_information = meteorology_properties.Extract_Meteorology_Properties(cdir,metadata,frac)
   
    #Assign all tile information
    #general = grp.create_group('general')
    for var in meteorology_information:
        mdgrp[var] = meteorology_information[var].astype(np.float64)
   
    #Assign the soil information
    print(nid,cid,"Assigning the soil information",memory_usage())
    if soil_frac > 0:
        soil = grp.create_group('soil')
        soil_information = information['soil']
        nsoil = soil_information['ntile']
        #Soil parameters
        soil['tile'] = np.int32(tiles[ttype == 3])
        for var in soil_information:
            if var in ['hidx_j']:
                soil[var] = np.int32(soil_information[var])
            elif var in ['hidx_k']:
                soil[var] = np.int32(soil_information[var])
            elif var == 'nsoil':
                soil[var] = np.int32(soil_information[var])
            else:
                soil[var] = np.float64(soil_information[var])
   
    #Assign the lake information 
    print(nid,cid,"Assigning the lake information",memory_usage())
    if lake_frac > 0:
        lake = grp.create_group('lake')
        lake_information = information['lake']
        nlake = lake_information['ntile']
        #Lake parameters
        lake['tile'] = np.int32(tiles[ttype==2])
        #output the variables
        for var in lake_information: 
            if var == 'nlake':
                lake[var] = np.int32(lake_information[var])
            elif var in ['refl_dry_dir','refl_dry_dif','refl_sat_dir','refl_sat_dif']:
                tmp = []
                for itile in range(nlake):
                    tmp.append(lake_information[var])
                lake[var] = np.float64(tmp).T
            else:
                lake[var] = np.float64(lake_information[var])*np.ones(nlake,dtype=np.float64)
      
    #Assign the glacier information
    print(nid,cid,"Assigning the glacier information",memory_usage())
    if glacier_frac > 0:
        glacier_information = information['glacier']
        glacier = grp.create_group('glacier')
        nglacier = glacier_information['ntile']
        #Glacier parameters
        glacier['tile'] = np.int32(tiles[ttype==1])
        #output the variables
        for var in glacier_information:
            if var == 'nglacier':
                glacier[var] = np.int32(glacier_information[var])
            elif var in ['refl_min_dir','refl_min_dif','refl_max_dir','refl_max_dif']:
                tmp = []
                for itile in range(nglacier):
                    tmp.append(glacier_information[var])
                glacier[var] = np.float64(tmp).T
            else:
                glacier[var] = np.float64(glacier_information[var])*np.ones(nglacier,dtype=np.float64)
      
    #Close file
    grp.close()
   
    #Print cid to node id log (beginning)
    os.system('echo end:%s >> %s/workspace/nid/%s.txt' % (cid,metadata['dir'],nid))
   
    return
   



def ezdev_compute_latlon_topo3dterrain(dem_latlon_tif,
                                      cdir,minx,miny,maxx,maxy,eares,lproj, log
                                      ):
    ##### EZDEV - COMPUTE LATLON TERRAIN PROPERTIES
    # demll = gdal_tools.read_raster(dem_latlon_tif)
    # md3['nodata'] = -9999.0
    demll = gdal_tools.read_raster(dem_latlon_tif).astype(np.float32)
    md3ll = gdal_tools.retrieve_metadata(dem_latlon_tif)
    md3ll['nodata'] = -9999.0
    maskll = gdal_tools.read_raster('%s/mask_buffered_latlon.tif' % cdir)
    demll[maskll == -9999.0] = -9999.0
    #Ensure all points are at or above sea level
    demll[(demll != -9999.0) & (demll < 0.0)] = 0.0
    # EZDEV added
    # ezdev_filename0 = os.path.join('/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/point11/land/tile:1,is:1,js:1',
    #                     'ezdev_dem_latlon.pkl')
    # ezdev_outfile0 = open(ezdev_filename0,'wb') 
    # pickle.dump(df, ezdev_outfile0)
    # ezdev_outfile0.close()
    ## end added 
    # EZDEV: compute slope and aspect from LAT/LON dataset
    # and then transformm to EA
    print('EZDEV: compute terrain vars from ll dem -------------------------------------')
    df_latlon_dx = (md3ll['maxx'] - md3ll['minx'])/md3ll['nx']
    df_latlon_dy = (md3ll['maxy'] - md3ll['miny'])/md3ll['ny']
    # dist = terrain_tools.calculate_distance(lat0,lat1,lon0,lon1)
    # use central lat lon in the future
    df_dx = terrain_tools.calculate_distance(md3ll['miny'], md3ll['miny'],                
                                             md3ll['minx'], md3ll['minx'] + df_latlon_dx )
    df_dy = terrain_tools.calculate_distance(md3ll['miny'], md3ll['miny'] + df_latlon_dy, 
                                             md3ll['minx'], md3ll['minx'] )

    df_dxy = np.sqrt(df_dx * df_dy)
    print(df_latlon_dx)
    print(df_latlon_dy)
    print(df_dx)
    print(df_dy)
    print(df_dxy)
    print(md3ll['maxx'])
    print(md3ll['minx'])
    print(md3ll['maxy'])
    print(md3ll['miny'])
    print(md3ll['nx'])
    print(md3ll['ny'])
    # df_latlon_dx = 60.0
    # df_latlon_dy = 90.0
    print('latlon map:: number of -9999 = {}'.format(np.size(demll[demll<0.0])))
    slope_ll, aspect_ll = gradient_d8(demll, df_dx, df_dy, aspect_rad=True)
    svf0_ll, tcf0_ll = viewf(demll.astype(np.float64), spacing=df_dxy, nangles=32)
    coss_ll = np.cos(slope_ll)
    sins_ll = np.sin(slope_ll)
    svf_ll = svf0_ll/coss_ll
    tcf_ll = tcf0_ll/coss_ll
    print(np.mean(aspect_ll))
    print(np.mean(slope_ll))
    print(np.mean(svf0_ll))
    print(np.mean(tcf0_ll))
    print(np.mean(svf_ll))
    print(np.mean(tcf_ll))
    # this convention for the topocalc slope / aspect
    sinssina_ll = - sins_ll/coss_ll * np.cos(aspect_ll) # slope => slope tangent
    sinscosa_ll = sins_ll/coss_ll * np.sin(aspect_ll) # slope => slope tangent

    # write rasters using the same metadata
    dem_llsvf_tif = '%s/llsvf_latlon.tif' % cdir
    dem_lltcf_tif = '%s/lltcf_latlon.tif' % cdir
    dem_llsinssina_tif = '%s/llsinssina_latlon.tif' % cdir
    dem_llsinscosa_tif = '%s/llsinscosa_latlon.tif' % cdir
    gdal_tools.write_raster(dem_llsvf_tif, md3ll, svf_ll)
    gdal_tools.write_raster(dem_lltcf_tif, md3ll, tcf_ll)
    # note here invert the definition to match the one from UCLA dataaset
    gdal_tools.write_raster(dem_llsinscosa_tif, md3ll, sinscosa_ll)
    gdal_tools.write_raster(dem_llsinssina_tif, md3ll, sinssina_ll)

    dem_easvf_tif = '%s/llsvf_ea.tif' % cdir
    dem_eatcf_tif = '%s/lltcf_ea.tif' % cdir
    dem_easinssina_tif = '%s/llsinssina_ea.tif' % cdir
    dem_easinscosa_tif = '%s/llsinscosa_ea.tif' % cdir
    # lproj = eaproj % float((maxlon+minlon)/2) # defined above, with minx, maxx ...
    # do not overwrite log in the future here ...
    os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (minx,miny,maxx,maxy,eares,eares,lproj, dem_llsvf_tif, dem_easvf_tif, log))
    os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (minx,miny,maxx,maxy,eares,eares,lproj, dem_lltcf_tif, dem_eatcf_tif, log))
    os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (minx,miny,maxx,maxy,eares,eares,lproj, dem_llsinscosa_tif, dem_easinscosa_tif, log))
    os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (minx,miny,maxx,maxy,eares,eares,lproj, dem_llsinssina_tif, dem_easinssina_tif, log))

    svf = gdal_tools.read_raster(dem_easvf_tif).astype(np.float32)
    tcf = gdal_tools.read_raster(dem_eatcf_tif).astype(np.float32)
    sinscosa = gdal_tools.read_raster(dem_easinscosa_tif).astype(np.float32)
    sinssina = gdal_tools.read_raster(dem_easinssina_tif).astype(np.float32)
    print('EZDEV: end compute terrain vars from ll dem ---------------------------------')
    return svf, tcf, sinscosa, sinssina
    #####


def compute_topo3d_properties(demns_topo3d, eares, topo3d_from_latlon): 
    """
    argument: dem in equal area with buffer
    return quantities of interest for 3D radiation parameterization
    """
    # -----------------------------------------------------------------------------
    # EZDEV: do topography analysis on data BEFORE removing PITS!
    # Calculate slope and aspect using Nate's code
    # demns_topo3d = rdem.data.copy()
    # print(nid,cid,'TA: Calculating slope and aspect', memory_usage())
    res_array = np.copy(demns_topo3d)
    res_array[:] = eares
    # THIS SHOULD BE DONE IN EA MAP, WITH ENOUGH BOUNDARIES (E.G., MOVE BUFFER TO 200 CELLS)
    # print('equal area map:: number of -9999 = {}'.format(np.size(demns_topo3d[demns_topo3d<0.0])))
    (slope,aspect) = terrain_tools.ttf.calculate_slope_and_aspect(demns_topo3d,res_array,res_array)


    slope_topocalc, aspect_topocalc = gradient_d8(demns_topo3d, eares, eares, aspect_rad=True)

    # print('EZDEV: computing cosine of slope (coss) and '
    #       'sin slope * aspect (sins*cosa, sins*sina)')

    coss = np.cos(slope_topocalc)
    sinslope = np.sin(slope_topocalc)

    # if not metadata['topo3d_from_latlon']:
    if not topo3d_from_latlon:
        # Let's not use Nate's definition here::
        # slopeangle_nate = np.arctan(slope)
        # coss_nate = np.cos(slopeangle_nate)
        # sinssina = sinslope/coss_nate * np.cos(aspect) # slope => slope tangent
        # sinscosa = sinslope/coss_nate * np.sin(aspect) # slope => slope tangent
        # IN NATE CONFIG, SLOPE IS ALREADY THE TANGENT
        use_Nate_slopeasp = False
        if use_Nate_slopeasp:
            sinssina = slope * np.cos(aspect)
            sinscosa = slope * np.sin(aspect)
        else:
            # ------------------------------------------------------------------------------------
            # TOPOCALC:: WHEN ASPECT RAD = True USES IPW RADIANS
            # aspect = 0 south, increases east to +pi, descreases west to -pi (North = +-pi)
            # to recover the UCLA definition (cosas -E, +W; sinas +N; - SOUTH)
            # we need the following change of coordinates:: 
            # ------------------------------------------------------------------------------------
            # USE UCLA TOPOGRAPHY CONVENTION
            sinssina = sinslope/coss * ( - np.cos(aspect_topocalc)) # POSITIVE NORTH
            sinscosa = sinslope/coss * ( + np.sin(aspect_topocalc)) # POSITIVE EAST
            # ------------------------------------------------------------------------------------



    # new convolution based computation of local stelev::
    # expensive calculation, do it only if really needed!
    ############ this is awfully slow for large convolution windows ###############
                        # 12 pix = ~ 1 km window size
                        # 24 pix = ~ 2 km window size
    """     aveblock_std = 24 # 100 pix = ~ 10 km window size
    # window = np.ones((aveblock_std, aveblock_std), dtype=np.float32)
    window = np.ones((aveblock_std, aveblock_std),
                            dtype=np.float32)/float(aveblock_std**2)
    mu = signal.convolve2d(demns, window, boundary='wrap', mode='same')
    sqdiff = np.sqrt((demns - mu) ** 2)
    radstelev = signal.convolve2d(sqdiff, window, boundary='wrap', mode='same')
    radavelev = mu # original dem data

    """
    # I am not using these variables so set them equal to the dem for now
    radavelev = demns_topo3d 
    
    # now computing the STANDARDIZED ELEVATION - to be used for clustering
    non_missing_data_mask = demns_topo3d > -9000.0
    elev_mean = np.mean(demns_topo3d[non_missing_data_mask])
    elev_std = np.std(demns_topo3d[non_missing_data_mask])
    if elev_std > 1E-6:
        # radstelev = (demns_topo3d - elev_mean)/elev_std
        radavelev = (demns_topo3d - elev_mean)/elev_std # normalized elevation
        radstelev = np.ones(np.shape(demns_topo3d))*elev_std
    else:
        radavelev = np.zeros(demns_topo3d.shape)
        radstelev = np.zeros(demns_topo3d.shape)
    radavelev[np.logical_not(non_missing_data_mask)] = -9999.0 # keep original mask

    # now here store the average standard deviation (same in each grid cell)
    # variable can be used for something else if need be
    radstelev[np.logical_not(non_missing_data_mask)] = -9999.0 # keep original mask

    #if not metadata['topo3d_from_latlon']:
    if not topo3d_from_latlon:
        svf_nangles = 16
        # svf_nangles = 32
        (svf0, tcf0) = viewf(demns_topo3d.astype(np.float64), spacing=eares, nangles=svf_nangles)
        svf = svf0/coss
        tcf = tcf0/coss
        # svf = svf0
        # tcf = tcf0
    else:    
        svf0 = np.zeros(svf.shape)*(-9999.0)
        tcf0 = np.zeros(svf.shape)*(-9999.0)

    # print('EZDEV: print some stats BEFORE cropping map')
    # print('EZDEV demns::')
    # # print('BUFFER = ', buffer)
    # print('shape = {}, min = {}, max = {}, mean = {}'.format(
    #             np.shape(demns_topo3d), np.min(demns_topo3d), np.max(demns_topo3d), np.mean(demns_topo3d)))
    # print('svf:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(svf), np.mean(svf), np.std(svf)))
    # print('tcf:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(tcf), np.mean(tcf), np.std(tcf)))
    # print('svf0:: max = {:.3f}, mean = {:.3f}, stdv = {:.3f}'.format(np.max(svf0), np.mean(svf0), np.std(svf0)))
    # print('tcf0:: max = {:.3f}, mean = {:.3f}, stdv = {:.3f}'.format(np.max(tcf0), np.mean(tcf0), np.std(tcf0)))
    # print('coss:: max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(np.max(coss), np.mean(coss), np.std(coss)))
    # print('sinscosa:: min = {:.3f}, max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(
    #             np.min(sinssina), np.max(sinscosa), np.mean(sinscosa), np.std(sinscosa)))
    # print('sinssina:: min = {:.3f}, max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(
    #             np.min(sinssina), np.max(sinssina), np.mean(sinssina), np.std(sinssina)))
    # print('radstelev::  min = {:.3f}, max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(
    #                 np.min(radstelev), np.max(radstelev), np.mean(radstelev), np.std(radstelev)))
    # print('demns_topo3d::  min = {:.3f}, max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(
    #                 np.min(demns_topo3d), np.max(demns_topo3d), np.mean(demns_topo3d), np.std(demns_topo3d)))
    # print('demns::  min = {:.3f}, max = {:.3f}, mean = {:.3f},  stdv = {:.3f}'.format(
                    # np.min(demns), np.max(demns), np.mean(demns), np.std(demns)))
    # end topo 3d ------------------------------------------------------------------------
    # exit()
    return slope, aspect, svf, tcf, sinscosa, sinssina, coss, radstelev, radavelev
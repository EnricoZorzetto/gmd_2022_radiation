#import netCDF4 as nc
import numpy as np
import os
import geospatialtools.gdal_tools as gdal_tools
import geospatialtools.terrain_tools as terrain_tools
import pickle
import collections
import psutil
import scipy.stats
import multiprocessing as mp
import scipy.stats
import scipy.sparse
import time
import pickle
from geospatialtools.terrain_tools import fwidth, frelief_inv, frelief
from geospatialtools import terrain_tools_fortran as ttf





def rp(target,args):
    p = mp.Process(target=target,args=args)
    p.start()
    p.join()
    return

def memory_usage():
    process = psutil.Process(os.getpid())
    #return process.memory_info().rss/1024/1024#process.memory_percent()
    return process.memory_percent()

def plot_data(data):
   
    import matplotlib.pyplot as plt
    data = np.ma.masked_array(data,data==-9999)
    print(np.unique(data))
    #vmin = np.mean(data) - np.std(data)
    #vmax = np.mean(data) + np.std(data)
    #plt.imshow(data,interpolation='nearest',vmin=vmin,vmax=vmax)
    plt.imshow(data,interpolation='nearest')
    plt.colorbar()
    plt.show()
    
    return

def normalize_variable(data):
   
    m = data != -9999
    if (np.max(data[m]) != np.min(data[m])):
        data[m] = (data[m] - np.min(data[m]))/(np.max(data[m]) - np.min(data[m]))
   
    return data

def gapfill_variable(data,type,value):
   
    m = data == -9999
    if type == 'd':
        if np.sum(~m) > 0:
            data[m] = value#scipy.stats.mode(data[~m])[0]
        else:
            data[m] = value
    if type == 'c':
        if np.sum(~m) > 0:
            data[m] = np.mean(data[~m])
        else:
            data[m] = value
    return data
   
def read_data(var,cdir,type,val,gf=True):

    data = gdal_tools.read_raster('%s/%s_ea.tif' % (cdir,var)).astype(np.float32)
    if gf == True:data = gapfill_variable(data,type,val)
    return data


def define_hillslopes(md,dem_file,slope_file,maxsmc_file,res,cdir,buffer,log,cid,eares):
    # POTENTIALLY RENAME define_hru
    
    #Define spatial resolution in meters
    #rdem = gdal_tools.read_data('%s/demns_latlon.tif' % cdir)
    rdem = gdal_tools.read_data('%s/demns_ea.tif' % cdir)
    #Calculate dx,dy,area
    #rdem = terrain_tools.calculate_area(rdem)
    #Define average spatial resolution (meters)
    res = eares #np.mean(rdem.area**0.5)
   
    #Clean up
    del rdem
   
    undef = -9999
    #Read in the data
    svf = read_data('svf',cdir,'c',10**-5) # EZDEV
    tcf = read_data('tcf',cdir,'c',10**-5) # EZDEV
    coss = read_data('coss',cdir,'c',10**-5) # EZDEV
    sinscosa = read_data('sinscosa',cdir,'c',10**-5) # EZDEV
    sinssina = read_data('sinssina',cdir,'c',10**-5) # EZDEV
    radstelev = read_data('radstelev',cdir,'c',10**-5) # EZDEV
    radavelev = read_data('radavelev',cdir,'c',10**-5) # EZDEV
    slope = read_data('slope',cdir,'c',10**-5)
    aspect = read_data('aspect',cdir,'c',10**-5)
    dem = read_data('demns',cdir,'c',0.0,gf=False)
    channels = read_data('channels',cdir,'d',0)
    hand = read_data('hand',cdir,'c',0)
    uhrt = read_data('uhrt_p2016',cdir,'c',1)
    uhst = read_data('uhst_p2016',cdir,'c',1)
    lt_uvt = read_data('lt_uvt_p2016',cdir,'c',1)
    ul_mask = read_data('ul_mask_p2016',cdir,'c',1) #upland/lowland mask
    #Set everything that is not upland to lowland
    ul_mask[(ul_mask != 1)] = 2
    hillslopes = read_data('hillslopes',cdir,'d',0,gf=False)
    maxsmc = read_data('ThetaS_0cm',cdir,'c',0.5)
    c2n = read_data('cultivated2natural',cdir,'d',1)
    g2t = read_data('grass2tree',cdir,'d',1)
    d2e = read_data('deciduous2evergreen',cdir,'d',0)
    c32c4 = read_data('c32c4',cdir,'d',0)
    cheight = read_data('cheight',cdir,'c',0)
    ksat = read_data('Ksat_0cm',cdir,'c',10**-4)
    irrigation = read_data('irrigation',cdir,'d',0)
    meteo_tas = read_data('tann',cdir,'c',273.15)
    meteo_prec = read_data('pann',cdir,'c',0.0)
    lats = np.linspace(0,1,dem.shape[0])
    lons = np.linspace(0,1,dem.shape[1])
    (longitude,latitude) = np.meshgrid(lons,lats)

    print('EZDEV debug hillslopes')
    print('shape input = {}'.format(np.shape(svf)))
    print('svf with -9999  :: min = {:.3f}, mean = {:.3f}, stdv = {:.3f}, max = {:.3f}'.format(np.min(svf),  np.mean(svf), np.std(svf), np.max(svf),))
    print('tcf with -9999  :: min = {:.3f}, mean = {:.3f}, stdv = {:.3f}, max = {:.3f}'.format(np.min(tcf),  np.mean(tcf), np.std(tcf), np.max(tcf),))
    print('svf without miss:: min = {:.3f}, mean = {:.3f}, stdv = {:.3f}, max = {:.3f}'.format(np.min(svf[svf > -1.0]),  np.mean(svf[svf > -1.0]), np.std(svf[svf > -1.0]), np.max(svf[svf > -1.0]),))
    print('tcf without miss:: min = {:.3f}, mean = {:.3f}, stdv = {:.3f}, max = {:.3f}'.format(np.min(tcf[tcf > -1.0]),  np.mean(tcf[tcf > -1.0]), np.std(tcf[tcf > -1.0]), np.max(tcf[tcf > -1.0]),))
   
    #Read in the metadata
    metadata = gdal_tools.retrieve_metadata(dem_file)
   
    #Ensure that the lakes and glaciers have been masked out
    mask = gdal_tools.read_raster('%s/mask_ea.tif' % cdir)
    lakes = gdal_tools.read_raster('%s/lakes_ea.tif' % cdir)
    glaciers = gdal_tools.read_raster('%s/glaciers_ea.tif' % cdir)

    # EZDEV - TOPO3D DEBUG - CHECK LAKES ANDGLACIER VALUES
    # lake_mask = mask.copy()
    # lake_mask[(lakes == 1)] = 1
    # lake_mask[(mask != 1) | (glaciers == 1)] = 0
    # print('lake svf = {}'.format( np.mean( svf[lake_mask == 1] )) )
    # print('lake ncells = {}'.format( np.size( svf[lake_mask == 1] )) )
    # glac_mask = mask.copy()
    # glac_mask[(glaciers == 1)] = 1
    # glac_mask[(mask != 1) | (lakes == 1)] = 0
    # print('glac svf = {}'.format( np.mean( svf[glac_mask == 1] )) )
    # print('glac ncells = {}'.format( np.size( svf[glac_mask == 1] )) )
    # # END TOPO3D DEUG

    # EZDEV: FIX THE CHANNELS -> MOVE TO NEIGHBOURING HILLSLOPES
    assign_channels_to_hills = False
    if assign_channels_to_hills:
        print('EZDEV: ASSIGN CHANNELS CELLS TO NEIGH HILLSLOPES')
        hillslopes = channels_2_hillslope_ezdev(hillslopes, dem, maxiter=5)


    # print('EZDEV DEBUG LAND-HILLSLOPES::')
    # print(' mask # < -1= {}, number .ne. 1 = {}'.format(mask.shape, np.size(mask[mask < 1.0 ])))
    # print(' mask # < -0= {}, number .ne. 1 = {}'.format(mask.shape, np.size(mask[mask < 0.0 ])))
    # print('hillslopes dem = {}, number -999   = {}'.format(dem.shape, np.size(dem[mask < -999 ])))
    # print('hillslopes hand = {}, number -999  = {}'.format(hand.shape, np.size(hand[mask < -999.0])))
    # print('hillslopes shape = {}, number -999 = {}'.format(hillslopes.shape, np.size(hillslopes[hillslopes < -900.0])))
    # print('svf        shape = {}, number -999 = {}'.format(svf.shape, np.size(svf[hillslopes < -900.0])))
    # print('hillslopes ksat = {}, number -999  = {}'.format(ksat.shape, np.size(ksat[mask < -999.0])))
    # print('hillslopes c2n  = {}, number -999  = {}'.format(c2n.shape, np.size(c2n[mask < -999.0])))
    # print('hillslopes meteo_tas  = {}, number -999  = {}'.format(meteo_tas.shape, np.size(meteo_tas[mask < -999.0])))
    # # print('svf pos hill = {}'.format( np.mean(svf[ np.logical_and(svf > -1.0, hillslopes > -1.0)])))
    # # print('svf neg hill = {}'.format( np.mean(svf[ np.logical_and(svf > -1.0, hillslopes < -1.0)])))
    # print('n channel cells = {}'.format( np.size( channels[channels > 0])))
    # neg_cond = np.logical_and( svf > 0.0, hillslopes < -9000.0)
    # pos_cond = np.logical_and( svf > 0.0, hillslopes > -9000.0)
    # nneg = np.size(neg_cond[neg_cond])
    # npos = np.size(pos_cond[pos_cond])
    # print('pos frac = {}, neg frac = {}'.format( nneg/(nneg + npos), npos/(nneg + npos)))
    # svf_pos = np.mean(svf[pos_cond])
    # svf_neg = np.mean(svf[neg_cond])
    # print('svf weight ave hill = {}'.format( (svf_neg*nneg + svf_pos*npos)/(nneg + npos) ))
    # print('svf average = {}'.format( np.mean(svf[svf > 0])))
    # ### and EZDEV
   

    #Set everything that is not soil to 0
    mask[(mask != 1) | (lakes == 1) | (glaciers == 1)] = 0
   
    #Define an updated mask
    tmask = (mask != 1) | (hillslopes == -9999) | (dem == -9999) | (hand == -9999) \
            | (meteo_tas == -9999) | (c2n == -9999) | (ksat == -9999)

    hillslopes[tmask] = undef
    dem[tmask] = undef
    hand[tmask] = undef
    slope[tmask] = undef
    aspect[tmask] = undef
    meteo_tas[tmask] = undef
    meteo_prec[tmask] = undef
    longitude[tmask] = undef
    latitude[tmask] = undef
   
    svf[tmask] = undef # EZDEV
    tcf[tmask] = undef # EZDEV
    coss[tmask] = undef  # EZDEV
    sinscosa[tmask] = undef # EZDEV
    sinssina[tmask] = undef  # EZDEV
    radstelev[tmask] = undef  # EZDEV
    radavelev[tmask] = undef  # EZDEV

    # hill_2_fix = np.logical_and(svf > 0.0, hillslopes < -0.0)
    # print('channels neg hill', channels[ np.logical_and( dem > 0, hillslopes < -0.0)])
    # print('nchannels neg hill', np.size(channels[ np.logical_and( dem > 0, hillslopes < -0.0)]))
    # print("-----")
    # print("EZDEV: preliminary value check 2")
    # print('mean svf = {}'.format(np.mean(svf[svf > -9000])))
    # print('# svf < -9999 = {}'.format(np.size(svf[svf < -9000.0])))
    # print('std svf = {}'.format(np.std(svf[svf > -9999.0])))
    # print('shape svf = {}'.format(np.shape(svf)))
    # print("-----")
   
    # import sys
    # sys.exit()
   
    #Calculate the hillslope properties
    print(cid,"Assembling the hillslope properties",
          '(Memory usage: %s percent)' % memory_usage())
    #hp_in = terrain_tools.calculate_hillslope_properties_updated(hillslopes,dem,
    #                      res,latitude,longitude,hand,slope,
    #                      aspect,meteo_tas,meteo_prec,cdir)
    # args = (hillslopes,dem,res,latitude,longitude,hand,slope,aspect,meteo_tas,meteo_prec,cdir,uhrt,uhst,
    #        lt_uvt,ul_mask) # EZDEV COMMENT
   
    args = (hillslopes,dem,res,latitude,longitude,hand,slope,aspect,
            svf, tcf, coss, sinscosa, sinssina, radstelev, radavelev,
            meteo_tas,meteo_prec,cdir,uhrt,uhst,
            lt_uvt,ul_mask) # EZDEV ADDED
   
    # property of each single hillslope elements before clustering  
    rp(calculate_hillslope_properties_ezdev, args) # EZDEV ADDED
    # rp(terrain_tools.calculate_hillslope_properties_updated,args) # EZDEV COMMENT
    hp_in = pickle.load(open('%s/hillslope_properties.pck' % cdir,'rb'))
   
    # print('hp_in = ', hp_in) # EZDEV
    print('hp_keys = ', hp_in.keys()) # EZDEV
    # print('hp_means = ', {key:np.mean(hp_in[key]) for key in hp_in.keys()}) # EZDEV
   
    #Clean up 
    #del latitude,longitude,meteo_tas,meteo_prec,aspect,dem,uhrt,uhst,lt_uvt,ul_mask
    # del latitude,longitude,meteo_tas,meteo_prec,aspect,uhrt,uhst,lt_uvt,ul_mask # EZDEV commented
    del latitude,longitude,meteo_tas,meteo_prec,uhrt,uhst,lt_uvt,ul_mask # EZDEV removed aspect
   
    print(cid,"Clustering the hillslopes",
               '(Memory usage: %s percent)' % memory_usage())
    #Assemble input data
    covariates = {}
    for var in md['hillslope']['hcov']:
        tmp = np.copy(hp_in[var])
        #Remove outliers
        # p5 = np.percentile(hp_in[var],0.01)
        # tmp[tmp < p5] = p5
        # p95 = np.percentile(hp_in[var],99.99)
        # tmp[tmp > p95] = p95
        # EZDEV:: do instead:
        p5 = np.min(hp_in[var])
        p95 = np.max(hp_in[var])
        if var in ['width_slope',]:
            tmp[(tmp >= -0.99) & (tmp < 0.0)] = tmp[(tmp >= -0.99) & (tmp < 0.0)]/0.99
            tmp[(tmp <= 99) & (tmp > 0.0)] = tmp[(tmp <= 99) & (tmp > 0.0)]/99
        covariates[var] = {'min':p5,
                           'max':p95,
                           't':-9999,
                           'd':tmp}
    #Compute the parameters for the clustering algorithm
    nc = md['hillslope']['k']

    #####
    # EZDEV: ALternatively, what Nate did in Yujin's case:
    #nc = md['hillslope']['k']
    # if md['hillslope']['k']['type'] == 'binning':
    #     # in this case the var is the dem (elevation only)
    #     nc = int(np.ceil((np.max(hp_in[var]) - np.min(hp_in[var]))/md['hillslope']['k']['dz']))
    #     nc = max(1,nc) #ensure it is non-zero
    # else:
    #     nc = md['hillslope']['k']['nc']
    #####
    #####    
    if 'use_adaptive_k' in md['hillslope'].keys(): 
        if md['hillslope']['use_adaptive_k'] == 1: 
            ncm = int(np.ceil((np.max(hp_in['dem']) - np.min(hp_in['dem']))/md['hillslope']['dz_adaptive_k']))
            print('modify k based on elevation')
            print('elevation dem: max = {}, min = {}, k dz = {}'.format(
                    np.max(hp_in['dem']), np.min(hp_in['dem']), md['hillslope']['dz_adaptive_k']))
            print('number of hillslopes changed from {} to {}'.format(nc, ncm))
            nc = ncm
            

    #####

    ws = np.ones(len(covariates.keys()))
    print(cid,'Clustering into %d characterstic hillslopes' % nc,
                           '(Memory usage: %s percent)' % memory_usage())
    dh = md['hillslope']['dh']
    max_nbands = md['hillslope']['max_nbands']
    min_nbands = md['hillslope']['min_nbands']
    #Compute the clusters
    (hillslopes_clusters,hp) = terrain_tools.cluster_hillslopes_updated(
                               hillslopes,covariates,hp_in,
                               nc,ws,dh,max_nbands,min_nbands)

    # EZDEV :: POSSIBLE ALTERNATIVE TO NORMALIZE USING STDV INSTEAD OF MIN - MAX
    # (hillslopes_clusters,hp) = cluster_hillslopes_ezdev(
    #                            hillslopes,covariates,hp_in,
    #                            nc,ws,dh,max_nbands,min_nbands)

    # EZDEV - for debugging only
    pickle.dump(hillslopes_clusters, open('%s/hillslopes_clusters_ezdev.pck' % cdir, 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(hp, open('%s/hp_ezdev.pck' % cdir, 'wb'), pickle.HIGHEST_PROTOCOL)
   
   
    #Mask out everything that is not land
    mask1 = mask #gdal_tools.read_raster('%s/mask_latlon.tif' % cdir)
    #mask2 = gdal_tools.read_raster('%s/soils_latlon.tif' % cdir)
    mask2 = gdal_tools.read_raster('%s/soils_ea.tif' % cdir)
    mask = (mask1 == 0) | (mask2 == -9999)
    #Clean up
    del mask1,mask2
    #Apply mask
    hillslopes_clusters[mask] = -9999
    channels[mask] = -9999
    hand[mask] = -9999
    slope[mask] = -9999
    maxsmc[mask] = -9999
    g2t[mask] = -9999
    c2n[mask] = -9999
    d2e[mask] = -9999
    c32c4[mask] = -9999
    cheight[mask] = -9999
    ksat[mask] = -9999
    irrigation[mask] = -9999
    svf[mask] = -9999 # EZDEV
    tcf[mask] = -9999 # EZDEV
    aspect[mask] = -9999 # EZDEV
    coss[mask] = -9999  # EZDEV
    sinscosa[mask] = -9999 # EZDEV
    sinssina[mask] = -9999  # EZDEV
    radstelev[mask] = -9999  # EZDEV
    radavelev[mask] = -9999  # EZDEV
   
   
   #Create the hillslope tiles (Associate a tile to each hru)
    print(cid,'Creating the height bands (dh = %f)' % dh,
                       '(Memory usage: %s percent)' % memory_usage())
    # HEIGHT BANDS
    (tiles,nhand) = terrain_tools.create_hillslope_tiles_updated(
                              hillslopes_clusters,hand,hillslopes,hp_in,hp)
   
    # print('partitioning hillslopes in height bands')
    # print('tiles=', tiles) # EZDEV
    # print('nhand=', nhand) # EZDEV
    # pickle.dump(tiles,open('%s/tiles_ezdev.pck' % cdir,'wb')) # EZDEV
    # exit()
   
    #Calculate the hrus (kmeans on each tile of each hillslope)
    tmp = {'maxsmc':maxsmc,'c2n':c2n,'g2t':g2t,'ksat':ksat,
                           'irr':irrigation,'d2e':d2e,'c32c4':c32c4,
           'sinscosa':sinscosa, 'sinssina':sinssina, 'svf':svf, 'tcf':tcf, # EZDEV added
           'radstelev':radstelev,
           'radavelev':radavelev,
           'cheight':cheight}
   
    # print('variables for HRUs tiles clustering') # EZDEV
    # print(md['hillslope']['tcov'])
    # print('those are the variables')
   
    covariates = {}
    for var in md['hillslope']['tcov']:
        covariates[var] = {
                        #    'min':md['hillslope']['tcov'][var]['min'], # EZDEV commented
                        #    'max':md['hillslope']['tcov'][var]['max'], # EZDEV commented
                           'min':np.min( tmp[var][tmp[var] > -9000 ]), # EZDEV added instead
                           'max':np.max( tmp[var][tmp[var] > -9000 ]), # EZDEV added instead
                           't':-9999,
                           'd':tmp[var]}
    ntiles = md['hillslope']['p']
    print(cid,"Clustering the height bands into %d tiles" % ntiles,
                      '(Memory usage: %s percent)' % memory_usage())
   
    ##### EZDEV ADDED: here istead of providing a fixed p, compute it from local elev stdv
    if 'use_adaptive_p' in md['hillslope'].keys():
        if md['hillslope']['use_adaptive_p'] == 1: # TRUE
            global_stelev = np.std(dem[dem > - 9000])
            print("EZDEV: here istead of providing fixed p compute it from local elev stdv = {}".format(global_stelev))
            print("grid cell wide standard deviation of elevation = {}".format(global_stelev))
            print('md[hillslope] keys:')
            print(md['hillslope'].keys())
            use_adaptive_p = bool(md['hillslope']['use_adaptive_p'])
            print('using adaptive p? {}'.format(use_adaptive_p))
            minh_adaptive_p = md['hillslope']['minh_adaptive_p']
            maxh_adaptive_p = md['hillslope']['maxh_adaptive_p']
            minp_adaptive_p = md['hillslope']['minp_adaptive_p']
            maxp_adaptive_p = md['hillslope']['maxp_adaptive_p']
            if global_stelev > maxh_adaptive_p:
                ntiles = maxp_adaptive_p   
            elif global_stelev < minh_adaptive_p:
                ntiles = minp_adaptive_p   
            else:
                ntiles = int(round( minp_adaptive_p + (maxp_adaptive_p - minp_adaptive_p)/(maxh_adaptive_p - minh_adaptive_p)*global_stelev ))
            print('based on local topography, ntiles (=p) has been changed to {}'.format(ntiles))
    ##### end EZDEV added    
   
    hrus = terrain_tools.create_hrus(hillslopes_clusters,tiles,covariates,ntiles,False,10,cdir)
    #args = (hillslopes_clusters,tiles,covariates,ntiles,False,10,cdir)
    #rp(terrain_tools.create_hrus,args)
    hrus = pickle.load(open('%s/hrus.pck' % cdir,'rb'))
   
    print('unique hrus = ', np.unique(hrus)) # EZDEV
    print('shape hrus = ', np.shape(hrus)) # EZDEV
    print('unique tiles = ', np.unique(tiles)) # EZDEV
    print('shape tiles = ', np.shape(tiles)) # EZDEV
   
    print(cid,"Calculating each hru's properties",'(Memory usage: %s percent)' % memory_usage())
    #hru_properties = terrain_tools.calculate_hru_properties_updated(hillslopes_clusters,tiles,
    #                res,hrus,hand,slope,hp)
    # args = (hillslopes_clusters,tiles,res,hrus,hand,slope,hp,cdir,nhand,dem) # EZDEV
    args = (hillslopes_clusters,tiles,res,hrus,hand,slope,hp,cdir,nhand,dem,
            aspect, svf, tcf, coss, sinscosa, sinssina, radstelev, radavelev) # EZDEV
    # rp(terrain_tools.calculate_hru_properties_updated,args)
    # print("-----")

    # EZDEV: should I move this after the channel filling? should solve the issue
    rp(calculate_hru_properties_ezdev, args) # EZDEV
    hru_properties = pickle.load(open('%s/hru_properties.pck' % cdir,'rb'))
   
   
    #Set the channels to be the most frequent surrounding hru
    print(cid,"Gap filling the hrus",'(Memory usage: %s percent)' % memory_usage())
    hrus = terrain_tools.ttf.gap_fill_hrus(hrus,channels)
   
    #Save the properties and the hru map
    print(cid,"Saving the output",'(Memory usage: %s percent)' % memory_usage())
    #pickle.dump(hru_properties,open('%s/hru_properties.pck' % cdir,'w'))
    metadata = gdal_tools.retrieve_metadata('%s/mask_ea.tif' % cdir)
    metadata['nodata'] = -9999.0
    #hrus
    gdal_tools.write_raster('%s/soil_tiles_ea.tif' % cdir,metadata,hrus)
   
    #Remap the hrus map to lat/lon
    mdll = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % cdir)
    minx = mdll['minx']
    maxx = mdll['maxx']
    miny = mdll['miny']
    maxy = mdll['maxy']
    res = np.abs(mdll['resx'])
    vars = ['soil_tiles',]
    for var in vars:
        os.system('rm -f %s' % '%s/%s_latlon.tif' % (cdir,var))
        os.system('gdalwarp -t_srs EPSG:4326 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f %s %s >& %s' % (res,res,minx,
                 miny,maxx,maxy,'%s/%s_ea.tif' % (cdir,var),
                 '%s/%s_latlon.tif' % (cdir,var),log))
   
    return hru_properties
   

def prepare_dem(lat,lon,metadata,eares):
   
    res = metadata['res']
    minlat = lat - res/2
    maxlat = lat + res/2
    minlon = lon - res/2
    maxlon = lon + res/2
    workspace = metadata['workspace']
    vrt = metadata['hillslope']['file']
   
    tmp = '%s/tmp.tif' % workspace
    tmp1 = '%s/tmp1.tif' % workspace
    dem_latlon = '%s/dem_latlon.tif' % workspace
   
    #1. Cutout the region of interest
    os.system('rm -f %s' % tmp)
    os.system('rm -f %s' % tmp1)
    os.system('gdalwarp -overwrite -te %.16f %.16f %.16f %.16f %s %s' % (minlon,minlat,maxlon,maxlat,vrt,tmp1))
    os.system('cp %s %s' % (tmp1,dem_latlon))
    os.system('gdalwarp -overwrite -r average -dstnodata -9999 -tr %.16f %.16f -t_srs EPSG:2163 %s %s' % (eares,eares,tmp1,tmp))
    # EZDEV changed average to lanczos - removed for now
   
    #Create the directory for the SAGA data
    workspace_saga = '%s/saga' % workspace
    os.system('rm -rf %s' % workspace_saga)
    os.system('mkdir -p %s' % workspace_saga)
   
    #2. Convert to SAGA format
    dem_sdat = '%s/dem.sdat' % workspace_saga
    dem = '%s/dem.sgrd' % workspace_saga
    os.system('rm -f %s' % dem_sdat)
    os.system("gdalwarp -of SAGA %s %s" % (tmp,dem_sdat))
   
    #3. Sink fill
    dem_ns = '%s/demns.sgrd' % workspace_saga
    os.system('saga_cmd ta_preprocessor 4 -ELEV %s -FILLED %s -MINSLOPE 0.01' % (dem,dem_ns))
   
    #4. Calculate slope
    slope = '%s/slope.sgrd' % workspace_saga
    os.system('saga_cmd ta_morphometry 0 -ELEVATION %s -SLOPE %s' % (dem_ns,slope))
   
    #5. Convert to tiff
    dem_ns_tif = '%s/dem_ns.tif' % workspace
    dem_ns_sdat = '%s/demns.sdat' % workspace_saga
    os.system('rm -f %s' % dem_ns_tif)
    os.system("gdalwarp -overwrite %s %s" % (dem_ns_sdat,dem_ns_tif))
    slope_tif = '%s/slope.tif' % workspace
    slope_sdat = '%s/slope.sdat' % workspace_saga
    os.system('rm -f %s' % slope_tif)
    os.system("gdalwarp -overwrite %s %s" % (slope_sdat,slope_tif))
   
    return (dem_ns_tif, slope_tif)


def Extract_Hillslope_Properties_NED(lat,lon,metadata):

 eares = 30.0
 #Extract the DEM for this cell
 (dem_file,slope_file) = prepare_dem(lat,lon,metadata,eares)
 #Extract the Porosity for this cell
 (maxsmc_file) = prepare_soil(lat,lon,metadata,eares)
 #Compute the hillslopes
 hru_properties = define_hillslopes(metadata,dem_file,slope_file,maxsmc_file,eares)
 #Prepare the properties for the model
 microtopo = 1.0E+036*np.ones(hru_properties['tile_id'].size)
 nt = len(hru_properties['tile_id'])
 output = {
           'hidx_j':hru_properties['tile_id'],
           'hidx_k':hru_properties['hillslope_id'],
           'microtopo':microtopo,
           'hlsp_length':hru_properties['hillslope_length'],#100*np.ones(nt)
           'hlsp_slope':hru_properties['slope'],#0.15*np.ones(nt)
           'hlsp_elev':hru_properties['depth2channel'],#0.15*np.cumsum(100*np.ones(nt))
           'hlsp_hpos':hru_properties['hillslope_position'],#np.linspace(50,950,10)
           'hlsp_width':(hru_properties['width_top']+hru_properties['width_bottom'])/2,#1.0*np.ones(nt)
          }

 #Convert to arrays
 for var in output:
  output[var] = np.array(output[var])

 return output

def Extract_Hillslope_Properties_HydroSheds(lat,lon,metadata):

 #Extract the DEM for this cell
 (dem_file,slope_file) = prepare_dem(lat,lon,metadata)
 #Extract the Porosity for this cell
 (maxsmc_file) = prepare_soil(lat,lon,metadata)
 #Compute the hillslopes
 hru_properties = define_hillslopes(metadata,dem_file,slope_file,maxsmc_file)
 #print hru_properties
 #Read in the hru properties
 #hru_properties = pickle.load(open('%s/hru_properties.pck' % metadata['workspace']))
 #Prepare the properties for the model
 microtopo = 1.0E+036*np.ones(hru_properties['tile_id'].size)
 output = {
           'hidx_j':hru_properties['tile_id'],
           'hidx_k':hru_properties['hillslope_id'],
           'microtopo':microtopo,
           'hlsp_length':hru_properties['hillslope_length'],           
           'hlsp_slope':hru_properties['slope'],
           'hlsp_elev':hru_properties['depth2channel'],
           'hlsp_hpos':hru_properties['hillslope_position'],
           'hlsp_width':hru_properties['width_top']/hru_properties['width_bottom'],
          }

 #Convert to arrays
 for var in output:
  output[var] = np.array(output[var])

 return output

def Extract_Hillslope_Properties_Original(metadata,soil_frac):

 import netCDF4 as nc
 #print NN
 NN = metadata['hillslope']['NN']
 #file = '/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/hillslope.nc'
 file = '/lustre/f1/unswept/Nathaniel.Chaney/data/misc/hillslope.nc'
 minlat = metadata['bbox'][2]
 minlon = metadata['bbox'][0]
 maxlat = metadata['bbox'][3]
 maxlon = metadata['bbox'][1]
 lat = (minlat+maxlat)/2
 lon = (minlon+maxlon)/2

 #Read in the soil type
 fp = nc.Dataset(file)
 lons = fp.variables['lon'][:]
 lats = fp.variables['lat'][:]

 #Change to -180 to 180 if necessary
 lons[lons > 180] = lons[lons > 180] - 360

 #Find the match
 amin = np.amin(np.abs(lats-lat))
 ilat = np.where(np.abs(lats-lat) == amin)[0][-1]
 amin = np.amin(np.abs(lons-lon))
 ilon = np.where(np.abs(lons-lon) == amin)[0][-1]

 #Extract the data
 vars = ['FRAC_TOPO_HLSPS','SOIL_E_DEPTH','MICROTOPO','HLSP_LENGTH','HLSP_SLOPE',
         'HLSP_SLOPE_EXP','HLSP_TOP_WIDTH','K_SAT_GW']
 frac_topo_hlsps = np.mean(fp.variables['FRAC_TOPO_HLSPS'][ilat,ilon])
 soil_e_depth = np.mean(fp.variables['SOIL_E_DEPTH'][ilat,ilon])
 microtopo = np.mean(fp.variables['MICROTOPO'][ilat,ilon])
 hlsp_length = np.mean(fp.variables['HLSP_LENGTH'][ilat,ilon])
 hlsp_slope = np.mean(fp.variables['HLSP_SLOPE'][ilat,ilon])
 hlsp_slope_exp = np.mean(fp.variables['HLSP_SLOPE_EXP'][ilat,ilon])
 hlsp_top_width = np.mean(fp.variables['HLSP_TOP_WIDTH'][ilat,ilon])
 k_sat_gw = np.mean(fp.variables['K_SAT_GW'][ilat,ilon])

 #Close the input file
 fp.close()

 #NN = 3 #Number of elevation tiles
 hidx_j = np.arange(NN)+1
 output = {
           'hidx_j':hidx_j,
           'hidx_k':[],
	   'microtopo':[],
           'tile_hlsp_length':[],
           'tile_hlsp_slope':[],
           'tile_hlsp_elev':[],
           'tile_hlsp_hpos':[],
	   'tile_hlsp_width':[],
           'tile_hlsp_frac':[],
           'frac':[],
          }
 #Process the tile information
 hk = 1
 for hj in hidx_j:
  output['microtopo'].append(microtopo)
  output['tile_hlsp_length'].append(hlsp_length/NN)
  rhj = float(hj)
  tile_hlsp_slope = hlsp_slope*((rhj/NN)**hlsp_slope_exp - ((rhj-1)/NN)**hlsp_slope_exp)/( rhj/NN - (rhj-1)/NN)
  output['tile_hlsp_slope'].append(tile_hlsp_slope)
  tile_hlsp_elev = hlsp_slope * hlsp_length * 0.5*((rhj/NN)**hlsp_slope_exp + ((rhj-1)/NN)**hlsp_slope_exp)
  output['tile_hlsp_elev'].append(tile_hlsp_elev)
  tile_hlsp_hpos = hlsp_length * (rhj-0.5)/NN
  output['tile_hlsp_hpos'].append(tile_hlsp_hpos)
  tile_hlsp_width = 1. + (rhj-0.5)/NN * (hlsp_top_width - 1.)
  output['tile_hlsp_width'].append(tile_hlsp_width)

 #Convert to arrays
 for var in output:
  output[var] = np.array(output[var])
 output['hidx_k'] = np.ones(output['hidx_j'].size)
 output['nsoil'] = output['tile_hlsp_width'].size
 tmp = np.ones(output['nsoil'])
 tmp = soil_frac*tmp/np.sum(tmp)
 output['frac'] = tmp
 output['tile_hlsp_frac'] = tmp
 #Define the number of tiles
 output['ntile'] = NN
  
 return output

def Extract_Hillslope_Properties(cdir,metadata,soil_frac,buffer,log):

 #Parameters
 #eares = metadata['fsres_meters']
 eares = -9999
 #Extract the DEM for this cell
 dem_file = '%s/dem_ns_ea.tif' % cdir
 slope_file = '%s/slope_ea.tif' % cdir
 maxsmc_file = '%s/ThetaS_ea.tif' % cdir
 #(dem_file,slope_file) = prepare_dem(lat,lon,metadata,eares)
 #Extract the porosity for this cell
 #(maxsmc_file) = prepare_soil(lat,lon,metadata,eares)
 #Compute the hillslopes
 hru_properties = define_hillslopes(metadata,dem_file,slope_file,maxsmc_file,eares,cdir,buffer,log)
 #print(hru_properties)
 #exit()
 #Prepare the properties for the model
 microtopo = 1.0E+036*np.ones(hru_properties['tile_id'].size)
 #hru_properties['hillslope_length'][:] = np.mean(hru_properties['hillslope_length'][:])
 #ls = hru_properties['hillslope_length'][:]
 #hru_properties['hillslope_position'][:] = np.cumsum(ls) - ls[0]/2 
 #hru_properties['width_top'][:] = np.mean(hru_properties['width_top'][:])
 #hru_properties['width_bottom'][:] = np.mean(hru_properties['width_bottom'][:])
 #hru_properties['slope'][:] = np.mean(hru_properties['slope'][:])
 #nt = hru_properties['tile_id'].size 
 width = ((hru_properties['width_top']+hru_properties['width_bottom'])/2).astype(np.float64)
 length = (hru_properties['hillslope_length']).astype(np.float64)
 wspec = hru_properties['wspec'][:]
 area = wspec*length
 frac = area/np.sum(area)
 output = {
           'hidx_j':hru_properties['tile_id'],
           'hidx_k':hru_properties['hillslope_id'],
           'microtopo':microtopo,
           'tile_hlsp_length':length,           
           'tile_hlsp_slope':hru_properties['slope'],
           'tile_hlsp_elev':hru_properties['depth2channel'],
           'tile_hlsp_hpos':hru_properties['hillslope_position'],
           'tile_hlsp_width':width,
           'twidth':hru_properties['width_top'],
           'bwidth':hru_properties['width_bottom'],
           'frac':area/np.sum(area)
           #'frac':hru_properties['area']/np.sum(hru_properties['area'])
          }

 #Convert to arrays
 for var in output:
  output[var] = np.array(output[var])#.astype(np.float32)
 #f0 = output['frac']
 #f1 = (output['tile_hlsp_length']*output['tile_hlsp_width'])/np.sum((output['tile_hlsp_length']*output['tile_hlsp_width']))
 #print (f1-f0)#.dtype

 '''#If results are weird then move to default
 if np.mean(output['tile_hlsp_length']) == 0:
   ntile = output['tile_hlsp_length'].size
   output['tile_hlsp_length'] = 100.0*np.ones(ntile)
   output['tile_hlsp_hpos'] = np.cumsum(output['tile_hlsp_length']) - output['tile_hlsp_length'][0]/2
   output['tile_hlsp_slope'] = 0.01*np.ones(ntile)
   output['tile_hlsp_elev'] = np.linspace(0,10,ntile)
   output['tile_hlsp_width'] = np.ones(ntile)
   output['tile_hlsp_width'] = np.ones(ntile)
   output['frac'] = np.ones(ntile)/np.sum(np.ones(ntile))'''

 #Make sure the frac is consistent
 output['frac'] = np.float64(soil_frac)*output['frac']

 #Define the number of tiles
 output['ntile'] = output['hidx_j'].size

 return output

def Extract_Hillslope_Properties_Updated(cdir,metadata,soil_frac,buffer,log,cid):

 #Parameters
 eares = metadata['fsres_meters']
 #eares = -9999
 #eares = 30 #CAREFUL
 #Extract the DEM for this cell
 dem_file = '%s/demns_ea.tif' % cdir
 slope_file = '%s/slope_ea.tif' % cdir
 maxsmc_file = '%s/ThetaS_ea.tif' % cdir
 #Compute the hillslopes
 hru_properties = define_hillslopes(metadata,dem_file,slope_file,
                                maxsmc_file,eares,cdir,buffer,log,cid,eares)

 #Prepare the properties for the model
 microtopo = 1.0E+036*np.ones(hru_properties['tile_id'].size)
 output = {
           'hidx_j':hru_properties['tile_id'],
           'hidx_k':hru_properties['hillslope_id'],
           'microtopo':microtopo,
     'tile_hlsp_aspect':hru_properties['hillslope_aspect'], # EZDEV
     'tile_hlsp_svf':hru_properties['hillslope_svf'], # EZDEV
     'tile_hlsp_tcf':hru_properties['hillslope_tcf'], # EZDEV
     'tile_hlsp_coss': hru_properties['hillslope_coss'],  # EZDEV
     'tile_hlsp_sinscosa':hru_properties['hillslope_sinscosa'], # EZDEV
     'tile_hlsp_sinssina':hru_properties['hillslope_sinssina'], # EZDEV
     'tile_hlsp_radstelev':hru_properties['hillslope_radstelev'], # EZDEV added
     'tile_hlsp_radavelev':hru_properties['hillslope_radavelev'], # EZDEV added
           'tile_hlsp_length':hru_properties['hillslope_length'],
           'tile_hlsp_slope':hru_properties['hillslope_slope'],
           'tile_hlsp_elev':hru_properties['hillslope_hand'],
           'tile_hlsp_hpos':hru_properties['hillslope_position'],
           'tile_hlsp_width':hru_properties['hillslope_width'],
           'tile_hlsp_frac':hru_properties['hillslope_frac'],
           'frac':hru_properties['frac'],
           'soil_depth':hru_properties['soil_depth'],
           'depth_to_bedrock':hru_properties['depth_to_bedrock'],
           'hand_ecdf':hru_properties['hand_ecdf'],
           'hand_bedges':hru_properties['hand_bedges'],
           'tile_elevation':hru_properties['dem']
          }

 #Convert to arrays
 for var in output:
  output[var] = np.array(output[var])#.astype(np.float32)
  if len(output[var].shape) == 2: output[var] = output[var].T

 #Make sure the frac is consistent
 output['frac'] = np.float64(soil_frac)*output['frac']

 #Define the number of tiles
 output['ntile'] = output['hidx_j'].size

 return output


def calculate_hillslope_properties_ezdev(hillslopes, dem, res, latitude,
                                         longitude, depth2channel, slope,
                                         aspect, svf, tcf,
                                         coss, sinscosa, sinssina,
                                         radstelev, radavelev,
                                         tas, prec,
                                         cdir, uhrt, uhst,
                                         lt_uvt, ul_mask):
    # Convert aspect to cartesian coordinates
    x_aspect = np.sin(aspect)
    y_aspect = np.cos(aspect)

    # print('mean of coss = {}'.format(np.nanmean(coss)))

    # Initialize properties dictionary
    vars = ['latitude', 'longitude', 'dem', 'aspect', 'tas', 'prec', 'slope',
            'width_intercept', 'width_slope',
            'length', 'area', 'd2c_array', 'position_array', 'width_array',
            'relief', 'x_aspect', 'y_aspect', 'hid', 'relief_a', 'relief_b',
            'uhrt', 'uhst', 'lt_uvt', 'ul_mask',
            'svf', 'tcf', 'coss', 'sinscosa', 'sinssina', 'radstelev', 'radavelev'] # EZDEV
    properties = {}
    for var in vars: properties[var] = []

    # Assemble masks
    tic = time.time()
    masks = {}
    for i in range(hillslopes.shape[0]):
        for j in range(hillslopes.shape[1]):
            h = hillslopes[i, j]
            if h == -9999: continue
            if h not in masks: masks[h] = []
            masks[h].append([i, j])
    for id in masks.keys():
        masks[id] = np.array(masks[id])

    # Iterate through each hillslope to calculate properties
    count = 0
    for uh in np.sort(list(masks.keys())):
        # tic = time.time()
        imin = np.min(masks[uh][:, 0])
        imax = np.max(masks[uh][:, 0])
        jmin = np.min(masks[uh][:, 1])
        jmax = np.max(masks[uh][:, 1])

        # Extract covariates for region
        shs = np.copy(hillslopes[imin:imax + 1, jmin:jmax + 1])
        sd2c = np.copy(depth2channel[imin:imax + 1, jmin:jmax + 1])
        sslope = np.copy(slope[imin:imax + 1, jmin:jmax + 1])

        # Bin the d2c
        m = shs == uh
        sd2c[~m] = -9999
        nc = min(25, np.ceil(np.sum(m) * res ** 2 / 8100.0))
        nc = min(nc, np.unique(sd2c[m]).size)
        if nc > 1:
            tmp = np.sort(sd2c[m])
            bin_edges = tmp[np.arange(0, tmp.size, np.int(
                np.ceil(float(tmp.size) / (nc + 1))))]
            tmp = np.digitize(sd2c[m], bin_edges)
            # X = sd2c[m]
            # X = X[:,np.newaxis]
            # model = sklearn.cluster.KMeans(n_clusters=nc,random_state=35799)
            # tmp = model.fit_predict(X)+1
        else:
            tmp = np.array([1, ])
        cls = np.copy(sd2c)
        cls[m] = tmp[:]

        # Reassign the d2c and create a new hillslope
        data = {'slope': [], 'd2c': [], 'area': []}
        hillslope = np.zeros(sd2c.shape).astype(np.int32)
        for cl in np.unique(tmp):
            m1 = cls == cl
            if np.sum(m1) == 0: continue
            # Calculate properties
            data['slope'].append(np.mean(sslope[m1]))
            data['d2c'].append(np.mean(sd2c[m1]))
            data['area'].append(res ** 2 * np.sum(m1))
            # Add id
            hillslope[m1] = cl

        # Sort data
        argsort = np.argsort(data['d2c'])
        for var in data:
            data[var] = np.array(data[var])[argsort]

        # Construct position and length arryas
        s = data['slope']
        d2c = data['d2c']
        length = []
        slopes = []
        hand = []
        position = []
        ns = []
        # Ensure there are no zero slopes
        s[s == 0] = 10 ** -4
        # Use the d2c as the vertices
        for i in range(data['d2c'].size):
            # if (i == data['d2c'].size):
            # l = (d2c[i-1]-r)/s[i-1]#/2
            # slp = s[i-1]
            # hand.append(r + l*slp/2)
            # r = r + l*slp
            # slopes.append(slp)
            # pos = pos + l/2
            # position.append(pos)
            if i == 0:
                l = d2c[i] / s[i]  # /2
                slp = s[i]
                hand.append(l * slp / 2)
                r = l * slp
                slopes.append(slp)
                pos = l / 2
                position.append(pos)
            else:
                l = (d2c[i] - r) / ((s[i] + s[i - 1]) / 2)
                slp = (s[i] + s[i - 1]) / 2
                hand.append(r + l * slp / 2)
                r = r + l * slp
                slopes.append((s[i] + s[i - 1]) / 2)
                pos = pos + l / 2
                position.append(pos)
            length.append(l)
        length = np.array(length)
        slopes = np.array(slopes)
        position = np.array(position)
        hand = np.array(hand)
        # Quality control
        if (np.min(length) == 0.0) or (np.max(hand) == 0.0):
            hand = np.array([0.5, 1.5])
            length = np.array([10.0, 10.0])
            slopes = np.array([0.1, 0.1])
            position = np.array([5.0, 15.0])
            data['area'] = np.array([900.0, 900.0])
        # Place data
        data['position'] = position
        data['length'] = length
        data['slope'] = slopes
        data['d2c'] = hand
        '''length = []
        pos0 = 0
        dtmp = 0
        for i in range(data['d2c'].size):
         if (data['d2c'].size == 1):
          ld = d2c[i]/s[i]/2
          lu = ld
          ns.append(s[i])
         elif (i == data['d2c'].size-1):
          ld = ((d2c[i]-d2c[i-1])/((s[i]+s[i-1])/2))/2
          lu = ld
          ns.append((s[i]+s[i-1])/2)
         elif i == 0:
          ld = d2c[i]/s[i]/2
          lu = ((d2c[i+1]-d2c[i])/((s[i+1]+s[i])/2))/2
          ns.append((ld*s[i]+lu*(s[i+1]+s[i]))/(ld + lu))
         else:
          ld = ((d2c[i] - d2c[i-1])/((s[i-1]+s[i])/2))/2
          lu = ((d2c[i+1] - d2c[i])/((s[i]+s[i+1])/2))/2
          ns.append((ld*(s[i-1]+s[i]) + lu*(s[i]+s[i+1]))/(ld + lu))
         #Ensure ld/lu are not 0 (HACK)
         if ld == 0:
          ld = res#1.0 #meter
          ns.append(0.001)
         if lu == 0:
          lu = res#1.0 #meter
          ns.append(0.001)
         dtmp += (ld+lu)*s[i]
         print ld*s[i],(ld+lu)*s[i],s[i],d2c[i],dtmp

         pos = pos0 + ld
         position.append(pos)
         pos0 = pos + lu
         length.append(ld+lu)'''
        # data['position'] = np.array(position)
        # data['length'] = np.array(length)
        # data['nslope'] = np.array(ns)

        # Calculate width
        data['width'] = data['area'] / data['length']

        # Fit line to width and depth2channel (slope is derivative of the second)
        position = np.array(
            [0., ] + list(data['position']) + [data['length'][-1] / 2, ])
        w = np.array(
            [data['width'][0], ] + list(data['width']) + [data['width'][-1], ])
        # s = np.array([0.0,]+list(data['slope']))#+[0.0,])
        # s = np.array([0.0,]+list(data['nslope'])+[0.0,])
        # d2c = np.array([0.0,]+list(data['d2c'])+[data['d2c'][-1]+data['slope'][-1]*data['length'][-1]/2,])
        d2c = np.array([0.0, ] + list(data['d2c']) + [data['d2c'][-1], ])
        relief = d2c[-1]
        # print data['length']
        # print np.mean(data['nslope'])*np.sum(data['length'])
        # print 't,ap',relief,np.sum(data['slope']*data['length'])
        # Normalize position,width,d2c
        position = position / np.sum(length)
        d2c = d2c / relief
        # w[w > 20] = 20
        if d2c.size == 3:
            # Width
            fw = [0, 1]
            # Slope
            # fs = [0,s[1]]
            # relief
            fr = [1.0, 1.0]
        else:
            weights = np.cos(
                np.linspace(-np.pi / 4, np.pi / 4, position.size - 2))
            weights = weights / np.sum(weights)
            # Width
            tmp = w / np.max(w)
            w[tmp > 100] = 100 * tmp[tmp > 100]  # Limit on width differences
            # popt, pcov = scipy.optimize.curve_fit(fwidth,position,w,bounds=([0.0,-1000],[10**4,1000]))
            z = np.polyfit(position[1:-1], w[1:-1], 1, w=weights)
            # fw = [popt[1]/popt[0],1]
            fw = [z[0] / z[1], 1]
            if fw[0] > 99: fw[0] = 99
            if fw[0] < -0.99: fw[0] = -0.99
            # Slope
            # z = np.polyfit(position[1:-1],s[1:-1],1,w=weights)
            # if z[0] < -1: z[0] = -1
            # if z[0] > 1: z[0] = 1
            # if z[1] < 0: z[1] = 0
            # if z[1] > 1: z[1] = 1
            # popt, pcov = scipy.optimize.curve_fit(fslope,position,s,bounds=([0.0,-1.0],[1.0,1.0]))
            # fs = [popt[1],popt[0]]
            # fs = [z[0],z[1]]
            # Relief
            # tic = time.time()
            if d2c[1:-1].size > 10:
                try:
                    fr, pcov = scipy.optimize.curve_fit(frelief, position[1:-1],
                                                        d2c[1:-1], bounds=(
                            [1.0, 1.0], [5.0, 5.0]))
                except:
                    fr = [1.0, 1.0]
            else:
                fr = [1.0, 1.0]

        tmp = {'latitude': latitude[imin:imax + 1, jmin:jmax + 1],
               'longitude': longitude[imin:imax + 1, jmin:jmax + 1],
               'dem': dem[imin:imax + 1, jmin:jmax + 1],
               'aspect': aspect[imin:imax + 1, jmin:jmax + 1],
               'tas': tas[imin:imax + 1, jmin:jmax + 1],
               'prec': prec[imin:imax + 1, jmin:jmax + 1],
               'slope': slope[imin:imax + 1, jmin:jmax + 1],
               'svf': svf[imin:imax + 1, jmin:jmax + 1],
               'tcf': tcf[imin:imax + 1, jmin:jmax + 1],
               'coss': coss[imin:imax + 1, jmin:jmax + 1],
               'sinscosa': sinscosa[imin:imax + 1, jmin:jmax + 1],
               'sinssina': sinssina[imin:imax + 1, jmin:jmax + 1],
               'radstelev': radstelev[imin:imax + 1, jmin:jmax + 1],
               'radavelev': radavelev[imin:imax + 1, jmin:jmax + 1],
               'x_aspect': x_aspect[imin:imax + 1, jmin:jmax + 1],
               'y_aspect': y_aspect[imin:imax + 1, jmin:jmax + 1],
               'uhrt': uhrt[imin:imax + 1, jmin:jmax + 1],
               'uhst': uhst[imin:imax + 1, jmin:jmax + 1],
               'lt_uvt': lt_uvt[imin:imax + 1, jmin:jmax + 1],
               'ul_mask': ul_mask[imin:imax + 1, jmin:jmax + 1]}

        # Add properties to dictionary
        for var in tmp:
            if np.sum(tmp[var] != -9999) > 0:
                properties[var].append(np.mean(tmp[var][tmp[var] != -9999]))
            else:
                properties[var].append(-9999)
        properties['width_intercept'].append(fw[1])
        # properties['slope_intercept'].append(fs[1])
        properties['width_slope'].append(fw[0])
        # properties['slope_slope'].append(fs[0])
        properties['relief_a'].append(fr[0])
        properties['relief_b'].append(fr[1])
        properties['length'].append(np.sum(data['length']))
        properties['area'].append(float(np.sum(data['area'])))
        properties['relief'].append(relief)
        properties['position_array'].append(position)
        properties['d2c_array'].append(d2c)
        properties['width_array'].append(w)
        properties['hid'].append(uh)
        length = np.sum(data['length'])
        # hslope = np.mean(tmp['slope'][m])#[tmp['slope'] != -9999])
        # fs = data['length']/length
        # print 'slope',hslope
        # print 'crelief',hslope*length
        # if count == 2:exit()
        count += 1

    # Finalize the properties
    for var in properties:
        # if var in ['position_array','width_array','d2c_array']:continue
        # properties[var] = np.array(properties[var])
        print('printing property = {}'.format(var)) # EZDEV
        print(len(properties[var])) # EZDEV
        # print([len(x1) for x1 in properties[var]])
        properties[var] = np.array(properties[var])

    # Sort by hid

    # return properties
    # Write out output
    pickle.dump(properties, open('%s/hillslope_properties.pck' % cdir, 'wb'),
                pickle.HIGHEST_PROTOCOL)
    return


def calculate_hru_properties_ezdev(hillslopes, tiles, res, hrus,
                                   depth2channel, slope, hp, cdir, nhand, dem,
                                   aspect, svf, tcf, coss,
                                   sinscosa, sinssina, radstelev, radavelev):
    # Get the hillslope fractions
    fs = []
    for ih in range(hp['hid'].size):
        hid = int(hp['hid'][ih])
        f = np.sum(hillslopes == hid) / float(hillslopes.size)
        # print('hillslope size in cells = {}'.format(np.sum(hillslopes==hid)))
        fs.append(f)
    fs = np.array(fs)
    hp['frac'] = fs / np.sum(fs)  # fix to 1

    # Assemble masks
    masks = {}
    for i in range(hillslopes.shape[0]):
        for j in range(hillslopes.shape[1]):
            hru = hrus[i, j]
            if hru == -9999.0: continue
            if hru not in masks: masks[hru] = []
            masks[hru].append([i, j])
    for hru in masks:
        masks[hru] = np.array(masks[hru])


    # Gather some general hru information
    hru_properties = {}
    vars = ['hillslope_id', 'tile_id', 'hru', 'area', 'hillslope_slope',
            'hillslope_aspect', 'hillslope_svf', 'hillslope_tcf',  # EZDEV
            'hillslope_coss', 'hillslope_sinscosa', 'hillslope_sinssina',
            'hillslope_radstelev', 'hillslope_radavelev', # EZDEV
            # EZDEV
            'hand_ecdf', 'hand_bedges', 'dem']
    for var in vars: hru_properties[var] = []
    for hru in np.sort(list(masks.keys())):
        iss = masks[hru][:, 0]
        jss = masks[hru][:, 1]

        # print('iss = ', iss)
        # print('jss = ', jss)
        hru_properties['dem'].append(int(np.mean(dem[iss, jss])))
        hru_properties['hillslope_id'].append(
            int(np.mean(hillslopes[iss, jss])))
        hru_properties['tile_id'].append(int(np.mean(tiles[iss, jss])))
        hru_properties['hru'].append(int(hru))
        hru_properties['area'].append(np.float64(res ** 2 * np.sum(iss.size)))
        hru_properties['hillslope_slope'].append(
            np.float64(np.mean(slope[iss, jss])))

        hru_properties['hillslope_aspect'].append(
            np.float64(np.mean(aspect[iss, jss])))
        hru_properties['hillslope_svf'].append(
            np.float64(np.mean(svf[iss, jss])))
        hru_properties['hillslope_tcf'].append(
            np.float64(np.mean(tcf[iss, jss])))
        hru_properties['hillslope_coss'].append(
            np.float64(np.mean(coss[iss, jss])))
        hru_properties['hillslope_sinscosa'].append(
            np.float64(np.mean(sinscosa[iss, jss])))
        hru_properties['hillslope_sinssina'].append(
            np.float64(np.mean(sinssina[iss, jss])))
        hru_properties['hillslope_radstelev'].append(
            np.float64(np.mean(radstelev[iss,jss])))
        hru_properties['hillslope_radavelev'].append(
            np.float64(np.mean(radavelev[iss,jss])))
        # Assign ecdf of hand - mean(hand) per hru
        # tmp = depth2channel[iss,jss]
        tmp = nhand[iss, jss]
        if np.sum(tmp != -9999) == 0:
            tmp[tmp == -9999] = 0.0
        else:
            tmp[tmp == -9999] = np.mean(tmp[tmp != -9999])
        # Compute the ecdf
        nbins = 10
        (hist, bin_edges) = np.histogram(tmp, bins=nbins)
        ecdf = np.cumsum(hist).astype(np.float32)
        ecdf = ecdf / ecdf[-1]
        ecdf = np.append(np.zeros(1), ecdf)
        hru_properties['hand_ecdf'].append(ecdf)
        hru_properties['hand_bedges'].append(bin_edges)
    print('EZDEV printing shape of hill arrays') # EZDEV
    for var in hru_properties:
        hru_properties[var] = np.array(hru_properties[var])
        print(np.shape(hru_properties[var])) # EZDEV
    # hru_properties['frac'] = np.zeros(hru_properties['area'].size) # EZDEV commented this
    # hru_properties['frac'] = np.zeros(hru_properties['area']/np.sum(hru_properties['area']) # EZDEV: this was already commented by Nate
    hru_properties['frac'] = hru_properties['area']/np.sum(hru_properties['area']) # EZDEV: let's do this instead 

    # Add fill for the other properties
    vars = ['hillslope_length', 'hillslope_hand', 'hillslope_position',
            'hillslope_width', 'hillslope_frac',
            'soil_depth', 'depth_to_bedrock']
    for var in vars:
        hru_properties[var] = np.zeros(hru_properties['area'].size).astype(
            np.float64)

    # Associate the hillslope properties
    for ih in range(hp['hid'].size):
        hid = int(hp['hid'][ih])
        # print hid
        m = hru_properties['hillslope_id'] == hid
        # Extract the tids
        (tids, idx) = np.unique(hru_properties['tile_id'][m],
                                return_inverse=True)
        # Compute the normalized relief
        nrelief = np.linspace(0, 1, 2 * tids.size + 1)[0::2]
        # Compute the correposponding lengths
        p0 = hp['relief_p0'][ih]
        p1 = hp['relief_p1'][ih]
        length = hp['length'][ih] * (
                    frelief_inv(nrelief[1:], p0, p1) - frelief_inv(
                nrelief[0:-1], p0, p1))
        # Compute the relief for each segment
        # hand = []
        # for i in range(nrelief.size-1):
        # x = np.linspace(nrelief[i],nrelief[i+1],100)
        # hand.append(np.mean(frelief(x,p0,p1)))
        # hand = hp['relief'][ih]*np.array(hand)
        hand = hp['relief'][ih] * (nrelief[0:-1] + nrelief[1:]) / 2
        # Compute the width for each segment
        pos = frelief_inv(nrelief, p0, p1)
        p0 = hp['width_p0'][ih]
        width = (fwidth(pos[1:], p0) + fwidth(pos[0:-1], p0)) / 2
        # Convert all to float64
        length = length.astype(np.float64)
        width = width.astype(np.float64)
        # Correct to the actual area (This is necessary for conservation in div_it...)
        # print idx
        # print length
        # print width
        # tmp = hru_properties['area'][m][idx]
        # tmp = np.sum(length*width)*tmp/np.sum(tmp)
        # r = tmp/(length*width)
        # length = r*length
        hand = hand.astype(np.float64)
        # Compute the fractions
        frac = (width * length) / np.sum(width * length)
        # Compute the positions
        positions = np.linspace(0, 1, 2 * tids.size + 1)[1::2]
        # Place the properties
        hru_properties['hillslope_length'][m] = length[idx]
        hru_properties['hillslope_hand'][m] = hand[idx]
        hru_properties['hillslope_position'][m] = positions[idx]
        hru_properties['hillslope_width'][m] = width[idx]
        # Compute the hillslope fraction
        for it in range(tids.size):
            m1 = m & (hru_properties['tile_id'] == tids[it])
            f = hru_properties['area'][m1] / np.sum(hru_properties['area'][m1])
            hru_properties['hillslope_frac'][m1] = frac[it] * f
        # Set the overall fraction
        # -------------------------------
        # EZDEV: comment the following method for computing the fraction of each hru
        # it introduces a bias for average terrain properties
        # EZDEV: Instead, see above
        # hru_properties['frac'][m] = hp['frac'][ih] * \
        #                             hru_properties['hillslope_frac'][m]
        # -------------------------------



        # print ih,tids.size
        # Determine if hillslope is in the lowlands or uplands (per pelletier 2016)
        if hp['ul_mask'][ih] >= 1.5:  # LOWLAND
            soil_thickness = 2.0
            sedimentary_thickness = hp['lt_uvt'][ih] - soil_thickness
            if sedimentary_thickness < 0: sedimentary_thickness = 0.0
            soil_depth = soil_thickness * np.ones(tids.size)
            depth_to_bedrock = (
                                           soil_thickness + sedimentary_thickness) * np.ones(
                tids.size)
        elif hp['ul_mask'][ih] < 1.5:  # UPLAND
            soil_thickness = np.linspace(2.0, hp['uhst'][ih], tids.size)
            regolith_thickness = np.linspace(hp['lt_uvt'][ih], hp['uhrt'][ih],
                                             tids.size)
            soil_depth = soil_thickness
            depth_to_bedrock = regolith_thickness  # soil_thickness + regolith_thickness
        hru_properties['soil_depth'][m] = soil_depth[idx]
        hru_properties['depth_to_bedrock'][m] = depth_to_bedrock[idx]

    # return hru_properties
    # Write out output
    pickle.dump(hru_properties, open('%s/hru_properties.pck' % cdir, 'wb'),
                pickle.HIGHEST_PROTOCOL)

    return


# # ported from terrain_tools (EZDEV)
def cluster_hillslopes_ezdev(hillslopes,covariates,hp_in,nclusters,ws,dh,max_nbands,min_nbands):

 #Add weights to covariates
 for var in covariates:
  covariates[var]['w'] = ws[list(covariates.keys()).index(var)]

 X = []
 for var in covariates:
  otmp = np.copy(covariates[var]['d'])
  otmp[(np.isnan(otmp) == 1) | (np.isinf(otmp) == 1)] = 0.0
  tmp = np.copy(otmp)
  #Normalize and apply weight
#   tmp = covariates[var]['w']*normalize_variable(tmp,covariates[var]['min'],covariates[var]['max'])
  # tmp = covariates[var]['w']*terrain_tools.normalize_variable(tmp,covariates[var]['min'],covariates[var]['max'])
  #tmp = covariates[var]['w']*(tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
  myminval = np.mean(tmp)
  print('lower boundsry clustering = {}'.format(myminval)) # EZDEV
  tmp = covariates[var]['w']*(tmp-np.mean(tmp))/np.std(tmp) # EZDEV
  X.append(tmp)
 X = np.array(X).T
 clusters = terrain_tools.cluster_data(X,nclusters)+1 # EZDEV added modulename
 #Clean up the hillslopes
 hillslopes = np.array(hillslopes,order='f').astype(np.int32)
 ttf.cleanup_hillslopes(hillslopes)
 #Assign the new ids to each hillslpe
 hillslopes_clusters = ttf.assign_clusters_to_hillslopes(hillslopes, clusters)
 #Determine the number of hillslopes per cluster
 uclusters = np.unique(clusters)
 #nhillslopes = []
 #for cluster in uclusters:
 # nhillslopes.append(np.sum(clusters == cluster))
 #nhillslopes = np.array(nhillslopes)

 #Compute the average value for each hillslope of each property
 hp_out = {}
 hp_out['hid'] = []
 for cluster in uclusters:
  hp_out['hid'].append(cluster)
  m = clusters == cluster
  #Compute fraction dependent on area of hillslope
  frac = hp_in['area'][m]/np.sum(hp_in['area'][m])
  for var in hp_in:
   if var in ['position_array','width_array','d2c_array','hid']:continue
   if var not in hp_out:hp_out[var] = []
   hp_out[var].append(np.sum(frac*hp_in[var][m]))
  #Calculate the fraction
  if 'frac' not in hp_out:hp_out['frac'] = []
  hp_out['frac'].append(np.sum(hp_in['area'][m])/np.sum(hp_in['area']))

 #Compute the average width and d2c function
 vars = ['relief_p0','relief_p1','width_p0','w','p','d']
 for var in vars:
  if var in ['w','p','d']:
   hp_out[var] = {}
  else:
   hp_out[var] = []
 for cluster in uclusters:
  '''ids = np.where(clusters == cluster)[0]
  d = []
  p = []
  w = []
  for id in ids:
   print id
   d = d + list(hp_in['d2c_array'][id])
   #w = w + (1 + list(hp_in['width_array'][id])
   w = w + list(1 + hp_in['position_array'][id]*hp_in['width_slope'][id])
   p = p + list(hp_in['position_array'][id])
  d = np.array(d)
  w = np.array(w)'''
  mc = np.where(clusters == cluster)[0]
  d = hp_in['d2c_array'][mc]
  #print hp_in['position_array'][mc].shape
  #print hp_in['width_slope'][mc].shape
  if len(hp_in['position_array'][mc].shape) > 1:
   w = 1 + hp_in['position_array'][mc,:]*hp_in['width_slope'][mc][:,np.newaxis]
  else:
   w = 1 + hp_in['position_array'][mc]*hp_in['width_slope'][mc]
  p = hp_in['position_array'][mc]
  p = np.concatenate(p)
  d = np.concatenate(d)
  w = np.concatenate(w)
  hp_out['p'][cluster-1] = p
  hp_out['d'][cluster-1] = d
  hp_out['w'][cluster-1] = w
  #Fit curve to d2c
  #fr, pcov = scipy.optimize.curve_fit(frelief,p,d)#,bounds=([0.0,-1000],[10**4,1000]))
  #try:
  try:
   fr, pcov = scipy.optimize.curve_fit(frelief,p,d,bounds=([1.0,1.0],[5.0,5.0]))
  except:
   fr = [1.0,1.0]
  hp_out['relief_p0'].append(fr[0])
  hp_out['relief_p1'].append(fr[1])
  #Fit line to width
  try:
   fw, pcov = scipy.optimize.curve_fit(fwidth,p,w,bounds=([-0.99,],[99,]))
  except:
   fw = [1.0,]
  hp_out['width_p0'].append(fw[0])
  #plt.plot(p,d,'bo',alpha=0.05)
  #plt.plot(p,frelief(p,fr[0],fr[1]),'ro',alpha=0.05)
  #plt.show()
  
 #Convert to arrays
 for var in hp_out:
  if var in ['p','d','w']:continue
  hp_out[var] = np.array(hp_out[var])

 #Define the number of elevation tiles per cluster
 tile_relief = dh#md['clustering']['tile_relief']
 max_ntiles = max_nbands#md['clustering']['max_ntiles']
 min_ntiles = min_nbands#md['clustering']['max_ntiles']
 nbins = np.round(hp_out['relief']/tile_relief).astype(np.int)
 nbins[nbins < min_ntiles] = min_ntiles
 nbins[nbins > max_ntiles] = max_ntiles
 hp_out['nbins'] = nbins

 #Set some constraints
 m = hp_out['length'] > 10000
 hp_out['length'][m] = 10000

 return (hillslopes_clusters,hp_out)


def get_pos_heighbours_ezdev(mat, dem, n0, n1, i, j):
    # neigh = []
    neigh0 = []
    neigh1 = []
    if i != 0:
        # neigh.append(mat[i-1, j])
        neigh0.append(i-1)
        neigh1.append(j)
    if i != n0 - 1:     
        # neigh.append(mat[i+1, j])
        neigh0.append(i+1)
        neigh1.append(j)
    if j != 0:
        # neigh.append(mat[i, j-1])
        neigh0.append(i)
        neigh1.append(j-1)
    if j != n1 - 1:     
        # neigh.append(mat[i, j+1])
        neigh0.append(i)
        neigh1.append(j+1)
    # neigh = [mat[id0, jd1] for (id0, jd1) in zip(neigh0, neigh1)]
    # nneigh = len(neigh0)
    # neigh = [ mat[el0, el1] for (el0, el1) in zip(neigh0, neigh1)]
    
    # print('i={}, j={}, neigh0={}, neigh1={}'.format(i, j, neigh0, neigh1))
    neigh0 = np.array(neigh0)
    neigh1 = np.array(neigh1)
    # neigh_val = np.array([ mat[neigh0[idx], neigh1[idx]] for idx in range(len(neigh0))])
    # neigh_dem = np.array([ dem[neigh0[idx], neigh1[idx]] for idx in range(len(neigh0))])

    neigh_val = mat[neigh0, neigh1]
    neigh_dem = dem[neigh0, neigh1]

    condkeep = neigh_val > 0

    pos_neigh_val = neigh_val[condkeep]
    pos_neigh_dem = neigh_dem[condkeep]
    pos_neigh_demdiff = np.abs(pos_neigh_dem - dem[i,j])

    # if np.size(pos_neigh_val) > 0:

    # demdiffs = [ np.abs(elen - dem[i.j]) for elen in neigh_dem]

    

    # print('neigh = {}'.format(neigh))
    # print('len neigh = {}'.format(len(neigh)))
    # posneigh = np.array([el for el in neigh_val if el > 0])
    # demneigh = np.array([el for el in neigh_res if el > 0])

    # if len(posneigh) > 0:
        # pick the value with closes elevation difference:
    # else:    
        # res = -9999
    # print('i={}, j={}, neigh0={}, neigh1={}'.format(i, j, neigh0, neigh1))
    # print('i={}, j={}, posneigh val={}'.format(i, j, posneigh))
    return pos_neigh_val, pos_neigh_demdiff


def channels_2_hillslope_ezdev(hillslopes, dem, maxiter=3, use_dem=False):
    """
    Given a map ("hillslopes") with positive integer values indentifying each hillslope 
    and -9999 for channels, assign each -9999 to one of the nearby hillslopes
    so as to remove channels completely and cover entire domain with hillslopes

    By default assigns the id of the channel cell to the 1st of the neighbous 
    (0th position in array of neighbours)
    if keep use_dem = False. If true, assign the id number of the cells with the closest
    elevation, but this also search updated cells so does not work well!

    maxiter ~ max channel width that gets filled

    # TO TEST THIS FUNCTION::
    (n0, n1) = (10, 10)
    A = np.ones((n0, n1))
    for i in range(n0):
        for j in range(n1):
            if i == j:
                A[i,j] = -1
            if i == j + 1:    
                A[i,j] = -1
            if i == j + 2:    
                A[i,j] = -1
            if i == j + 2:    
                A[i,j] = -1
            if i < j:    
                A[i,j] = 2
            if i == n0-j:
                A[i,j] = -1
            if i < n0-j and i < j:
                A[i,j] = 3
    DEM = A.copy()
    B = channels_2_hillslope_ezdev(A, DEM)
    plt.figure()
    plt.imshow(B)
    plt.colorbar()
    plt.show()

    """
    (n0, n1) = hillslopes.shape
    # fillhills = -9999*np.ones((n0, n1), dtype = int)
    fillhills = hillslopes.copy()
    # print(fillhills[0,0])
    # fillhills[fillhills < 0] = 3
    # do one pass and remove all -9999 with a non-neg neighbour above / below / r / l
    p0, p1 = np.where(fillhills < 0)
    nnegs = np.size(p0)
    itercount = 0
    # maxiter = 5
    while (nnegs > 0 and itercount < maxiter):
        print('iteration = {}'.format(itercount))
        print('number of negative points = {}'.format(nnegs))
        # p0, p1 = np.where(fillhills < 0)
        for indxp in range(nnegs):
            i = p0[indxp]
            j = p1[indxp]
            # mind the order? which one should I assign to? use the DEM?
            posval, posdemdiff = get_pos_heighbours_ezdev(fillhills, dem, n0, n1, i, j)
            if len(posval)>0:
                if use_dem: # get that with closest elevation instead
                    mindiffpos = np.argmin(posdemdiff)
                    fillhills[i,j] = posval[mindiffpos]
                else: # just get one of the neighbours
                    fillhills[i,j] = posval[0]
        # update counter and condition with updated fillhills array
        itercount += 1
        p0, p1 = np.where(fillhills < 0)
        nnegs = np.size(p0)
        print('nnegs = {}'.format(nnegs))

    print("Is there any negative value left? {}".format( np.any(fillhills < 0)))
    return fillhills
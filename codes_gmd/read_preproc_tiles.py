
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geospatialtools import gdal_tools
import demfuncs as dem
import gridfuncs as gf
import matplotlib


########################## my parameters #################################
domains = ['EastAlps, Nepal', 'Peru']
npvalues = [1, 2, 5, 10, 20, 100, 200]
# my_input_kind = 'k2n1pV'; # 2 hillslopes, 1 HB, variable # of tiles
my_input_kind = 'kVn1p5'; # 2 hillslopes, 1 HB, variable # of tiles
myk=my_input_kind[1]; myn=my_input_kind[3]; myp = my_input_kind[5]
# mydom = 'EastAlps'
mydom = 'EastAlps'
# myp = 100
myk = 50
mycosz = 0.3
myazi = 0.0*np.pi
myalbedo = 0.3
########################################################################

dem.matplotlib_update_settings()
# read tiling maps from GFDL_preprocessing codes
datadir = os.path.join('/Users/ez6263/Documents/gmd_2021_grids_light')
input_kinds =[f for f in os.listdir(datadir) if not f.startswith('.')]
ftilename = 'res_{}_k_{}_n_{}_p_{}'.format(mydom, myk, myn, myp)
myexpdir = os.path.join(datadir, my_input_kind, ftilename)
landdir = os.path.join(myexpdir, 'land', 'tile:1,is:1,js:1')



dbfile = os.path.join(myexpdir, 'ptiles.{}.tile1.h5'.format(ftilename))
# gtd = gf.grid_tile_database(dbfile)
gtd = gf.grid_tile_database(dbfile, landdir=landdir, tiles2map=True) # include high res maps

# gtd.tile
# gtd.ntiles

###### predict average ucla fluxes - grid average [SCALAR]
ucla_terrain = gf.ucla_terrain(domain=mydom)
AVE_ucla_pred = gf.prediction(svf=ucla_terrain.svf, tcf = ucla_terrain.tcf,
                              ss = ucla_terrain.ss, sc=ucla_terrain.sc,
                              type='lee', cosz=mycosz, azi=myazi, albedo=myalbedo)

###### predict tile-by-tile fluxes [1D ARRAY]
tiles_pred = gf.prediction(svf=gtd.svf, tcf = gtd.tcf, ss = gtd.ss, sc=gtd.sc,
                           type='lee', cosz=mycosz, azi=myazi, albedo=myalbedo)
###### AVERAGE the tile-by-tile fluxes AFTER PREDICTION [SCALAR]
tiles_pred.AVEfdir = np.sum(tiles_pred.fdir*gtd.frac)
tiles_pred.AVEfdif = np.sum(tiles_pred.fdif*gtd.frac)
tiles_pred.AVEfrdir = np.sum(tiles_pred.frdir*gtd.frac)
tiles_pred.AVEfrdif = np.sum(tiles_pred.frdif*gtd.frac)
####### predict tile-averaged fluxes [SCALAR] (First average terrain vars over tiles, then predict fluxes)
AVE_tile_pred = gf.prediction(svf=gtd.ave_svf, tcf = gtd.ave_tcf, ss = gtd.ave_ss, sc=gtd.ave_sc,
                         type='lee', cosz=mycosz, azi=myazi, albedo=myalbedo)

###### predict high res map fluxes [2D MAPS] - add option here to compress them to 1D arrays?
hrmap_pred = gf.prediction(svf=gtd.svf_hr_map.data, tcf = gtd.tcf_hr_map.data,
                           ss = gtd.ss_hr_map.data, sc=gtd.sc_hr_map.data,
                             type='lee', cosz=mycosz, azi=myazi, albedo=myalbedo)
###### now, after perdiction average fluxes
hrmap_pred.AVEfdir =  np.mean(hrmap_pred.fdir )
hrmap_pred.AVEfdif =  np.mean(hrmap_pred.fdif )
hrmap_pred.AVEfrdir = np.mean(hrmap_pred.frdir)
hrmap_pred.AVEfrdif = np.mean(hrmap_pred.frdif)
####### predict tile-averaged fluxes [SCALAR] (First average tiles, then predict fluxes)
AVE_hr_pred = gf.prediction(svf=gtd.svf_hr_ave, tcf = gtd.tcf_hr_ave, ss = gtd.ss_hr_ave, sc=gtd.sc_hr_ave,
                              type='lee', cosz=mycosz, azi=myazi, albedo=myalbedo)




# TO MAP TILES TO TERRAIN, THIS CAN BE DONE IN BOTH CLASSES::
# tiles_pred.fdir_mapped = gf.map_tiled_prop(gtd.tile, tiles_pred.fdir, gtd.tiles_hr_map.data)















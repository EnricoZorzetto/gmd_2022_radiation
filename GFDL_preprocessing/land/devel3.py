import numpy as np
import h5py
import os
import netCDF4 as nc
from geospatialtools import gdal_tools
import pickle


tid = 3
# lfile = os.path.join("/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/global_20220302", "ptiles.global_20220302.tile{}.h5".format(tid))
# lfile = os.path.join("/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/global_20220317", "ptiles.global_20220317.tile{}.h5".format(tid))
# lfile = os.path.join("/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/global1_640", "ptiles.global1_640.tile{}.h5".format(tid))
lfile = os.path.join("/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/global_20220319b", "ptiles.global_20220319b.tile{}.h5".format(tid))
# lfile = os.path.join("/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/res_cont_Africap", "ptiles.res_cont_Africap.tile{}.h5".format(tid))
# lfile = os.path.join("/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/res_cont_TPL", "ptiles.res_cont_TPL.tile{}.h5".format(tid))
hfile = "/lustre/f2/dev/Nathaniel.Chaney/data/hydrography/river_data_hydrography.c96_OM4_05-v20151028_20161005/river_data.tile{}.nc".format(tid)

# landcount = 0
# for mytid in range(1,4):
#     mylfile = os.path.join("/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/global1_640", "ptiles.global1_640.tile{}.h5".format(mytid))
#     myfp = h5py.File(mylfile)
#     mycount = len(myfp['grid_data'].keys())
#     landcount += mycount
#     print(mytid, mycount)
# print(landcount)

#parameters
species = {0:'C4GRASS',1:'C3GRASS',2:'TEMPDEC',3:'TROPICAL',4:'EVERGR'}
landuse = {1:'PAST',2:'CROP',3:'NTRL',4:'SCND',5:'URBN'}

ALL0 = []
ALL1 = []


# fp = h5py.File(lfile)

#Read in the metadata information from the hydrography file
fp = nc.Dataset(hfile)
lats = fp['grid_y'][:]
lons = fp['grid_x'][:]
nlon = lons.size
nlat = lats.size
data = {}

#Iterate through the cells
fp = h5py.File(lfile)

# print(len(fp['grid_data'].keys()))

count = 0
for cell in fp['grid_data']:
    count += 1
    # print(count,cell)
    #if (tid != 1) & (count <= 1066):continue
    #if cell == 'tile:3,is:5,js:20':continue
    #if cell != 'tile:2,is:91,js:20':continue
    #if cell == 'tile:1,is:12,js:5':continue
    #if cell == 'tile:1,is:12,js:91':continue
    #if cell == 'tile:1,is:13,js:2':continue
    #if cell != 'tile:1,is:29,js:92':continue
    #if cell == 'tile:6,is:38,js:38':continue
    #if cell == 'tile:6,is:63,js:44':continue
    #if cell == 'tile:6,is:64,js:44':continue
    #if cell != 'tile:6,is:38,js:38':continue
    gp = fp['grid_data'][cell]
    ilat = int(cell.split(',')[1].split(':')[1])-1
    ilon = int(cell.split(',')[2].split(':')[1])-1
    #Extract the desired information
    lat = gp['metadata']['latitude'][:][0]
    lon = gp['metadata']['longitude'][:][0]
    frac = np.array(gp['metadata']['frac'],ndmin=1)
    #print cell,lat,lon,ilat,ilon
    #Compute the standard deviation of the meteorological weights
    #vars = ['prec','srad','tavg','vapr','wind']
    vars = ['prec','srad','tavg','vapr','wind']
    for var in vars:
        mtmp = np.mean(np.array(gp['metadata'][var][:],ndmin=2),axis=0)
        '''if mtmp.size != frac.size:
            tmp0 = np.ones(frac.size)
            if mtmp.size > 0:
            tmp0[-mtmp.size:] = mtmp
            mtmp = tmp0'''
        if var not in data:
            data[var] = np.zeros((nlat, nlon))
            data[var][:] = -9999.0
        tmp0 = np.sum(frac*mtmp)
        tmp1 = np.sum(frac*(mtmp-tmp0)**2)
        data[var][ilat,ilon] = tmp1
    #Define the number of tiles
    ntiles = frac.size
    if 'ntiles' not in data:
        data['ntiles'] = np.zeros((nlat, nlon))
        data['ntiles'][:] = -9999.0
    data['ntiles'][ilat,ilon] = ntiles
    #Lake info
    if 'lake' in gp:
        lfrac = np.array(gp['lake']['frac'],ndmin=1)
        #Define the fraction of land that is lake
        if 'lfrac' not in data:
            data['lfrac'] = np.zeros((nlat,nlon))
            data['lfrac'][:] = -9999.0
        data['lfrac'][ilat,ilon] = np.sum(lfrac)
        # print('EZDEV: lake_frac lfrac = {}, sum = {}'.format(lfrac, np.sum(lfrac)))
        #Glacier info

        LAK = np.array(gp['lake']['sinssina'])
        if np.isnan(LAK) or LAK < -9000.0:
            print('latlon', lat, lon)
            print('LAK', LAK)

    if 'glacier' in gp:
        gfrac = np.array(gp['glacier']['frac'],ndmin=1)
        #Define the fraction of land that is lake
        if 'gfrac' not in data:
            data['gfrac'] = np.zeros((nlat, nlon))
            data['gfrac'][:] = -9999.0
        data['gfrac'][ilat,ilon] = np.sum(gfrac)

        GLA = np.array(gp['glacier']['sinssina'])
        if np.isnan(GLA) or GLA < -9000.0:
            print('latlon', lat, lon)
            print('GLA', GLA)

        # print('EZDEV: glac_frac gfrac = {}, sum = {}'.format(gfrac, np.sum(gfrac)))
    #Soil info
    if 'soil' in gp:
        sfrac = np.array(gp['soil']['frac'],ndmin=1)
        #Define the fraction of land that is soil
        if 'sfrac' not in data:
            data['sfrac'] = np.zeros((nlat,nlon))
            data['sfrac'][:] = -9999.0
        data['sfrac'][ilat,ilon] = np.sum(sfrac)
        # print('EZDEV: soil_frac sfrac = {}, sum = {}'.format(sfrac, np.sum(sfrac)))
        #Normalize the soil fraction
        sfrac = sfrac/np.sum(sfrac)
        #Find means
        SOI = np.array(gp['soil']['tile_hlsp_sinssina'])
        condsoi = np.logical_or( np.isnan(SOI), SOI < -9000.0)
        SOIp = SOI[condsoi]
        if np.size(SOIp) > 0:
            print('latlon', lat, lon)
            print('SOI', SOIp)



        # print(var, np.shape(sfrac), np.shape(np.array(gp['soil']['dat_chb'])), 
                                    # np.shape(np.array(gp['soil']['tile_hlsp_sinssina'])))
        #for var in ['dat_chb','dat_k_sat_ref','dat_w_sat','tile_hlsp_elev','tile_hlsp_length','tile_hlsp_slope','tile_hlsp_width','irrigation','dat_psi_sat_ref','bl','bsw','bwood','br']:
        # if np.size(sfrac) != np.size(np.array(gp['soil']['dat_chb'])):
        #     print(np.shape(sfrac), np.shape(np.array(gp['soil']['dat_chb'])), 
        #                         np.shape(np.array(gp['soil']['tile_hlsp_sinssina'])))

        # print(np.array(gp['metadata']['longitude']), np.array(gp['metadata']['latitude']))
        # gp['soil']['tile'][:]
        # gp['soil']['hidx_j'][:] # tile index
        # gp['soil']['hidx_k'][:] # hillslope index

        for var in ['dat_chb','dat_k_sat_ref','dat_w_sat','irrigation','dat_psi_sat_ref','bl','bsw',
                        'bwood','br','tile_hlsp_slope','tile_hlsp_width','wtd',
                        'soil_depth','depth_to_bedrock','gw_perm','ksat_0cm','ksat_200cm']:
            if np.size(sfrac) != np.size(np.array(gp['soil'][var])):
                print(var, np.shape(sfrac), np.shape(np.array(gp['soil']['dat_chb'])), 
                                    np.shape(np.array(gp['soil']['tile_hlsp_sinssina'])))
                raise Exception("!")

        #     tmp = np.sum(sfrac*np.array(gp['soil'][var],ndmin=1))
        #     if var not in data:
        #         data[var] = np.zeros((nlat, nlon))
        #         data[var][:] = -9999.0
        #     data[var][ilat,ilon] = tmp
        # #Compute hillslope properties
        # vars = ['tile_hlsp_length','tile_hlsp_elev']
        # for var in vars:
        #     if var not in data:
        #         data[var] = np.zeros((nlat, nlon))
        #         data[var][:] = -9999.0
        # hidx_k = gp['soil']['hidx_k'][:]
        # hidx_j = gp['soil']['hidx_j'][:]
        # ls = []
        # es = []
        # fs = []
        # for hid in np.unique(hidx_k):
        #     mk = hidx_k == hid
        #     l = 0
        #     e = 0
        #     f = 0
        #     for tid in np.unique(hidx_j[mk]):
        #         m = hidx_j[mk] == tid
        #         #print np.sum(m),np.sum(mk)
        #         l = l + np.mean(np.array(gp['soil']['tile_hlsp_length'],ndmin=1)[mk][m])
        #         f = f + np.sum(np.array(gp['soil']['frac'],ndmin=1)[mk][m])
        #         if tid == np.max(hidx_j[mk]):e = np.mean(np.array(gp['soil']['tile_hlsp_elev'],ndmin=1)[mk][m])
        #     ls.append(l)
        #     es.append(e)
        #     fs.append(f)
        # fs = np.array(fs)
        # fs = fs/np.sum(fs)
        # es = np.sum(fs*es)
        # ls = np.sum(fs*ls)
        # data['tile_hlsp_length'][ilat, ilon] = ls
        # data['tile_hlsp_elev'][ilat, ilon] = es
        # #Create species fractions
        # tmp = np.array(gp['soil']['vegn'],ndmin=1)
        # for isp in species:
        #     m = tmp == isp
        #     f = np.sum(sfrac[m])
        #     if species[isp] not in data:
        #         data[species[isp]] = np.zeros((nlat, nlon))
        #         data[species[isp]][:] = -9999.0
        #     data[species[isp]][ilat, ilon] = f
        # #Create landuse fractions 
        # tmp = np.array(gp['soil']['landuse'],ndmin=1)
        # for ilu in landuse:
        #     m = tmp == ilu
        #     f = np.sum(sfrac[m])
        #     if landuse[ilu] not in data:
        #         data[landuse[ilu]] = np.zeros((nlat, nlon))
        #         data[landuse[ilu]][:] = -9999.0
        #     data[landuse[ilu]][ilat, ilon] = f

    # sk = list(gp['soil'].keys())
    # print(sk)

    # for el in sk:
    #     shape = gp['soil'][el].shape
    #     # print(elem)
    #     # if not np.isscalar(elem):
    #     print(el, shape)

# # fp.close()


# print( np.array(gp['soil']['frac']) )
# print( np.array(gp['metadata']['latitude']) )
# print( np.array(gp['metadata']['longitude']) )
# print( np.array(gp['soil']['tile_hlsp_svf']) )

# # cid = 'tile:%d,is:%d,js:%d' % (tid,y,x)
# cid = 'tile:%d,is:%d,js:%d' % (tid,ilat+1,ilon+1)
# tfile = os.path.join("/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/global_20220318/land/{}".format(cid))
# hrus = gdal_tools.read_raster('%s/soil_tiles_ea.tif' % tfile)
# # hrus = gdal_tools.read_raster('%s/soil_tiles_latlon.tif' % tfile)
# # hruss = gdal_tools.read_raster('%s/soil_tiles_ea.tif' % tfile)

# print( np.unique(hrus) )
# print( np.shape(hrus) )


# # with open( os.path.join(tfile, 'hrus.pck'),'rb') as file:
# with open( os.path.join(tfile, 'hru_properties.pck'),'rb') as file:
#     hrusp = pickle.load(file)

# # hrus1 = hrus[300:np.shape(hrus)[0]-300, 300:np.shape(hrus)[1]-300]
# print(np.unique(hrusp))
# print(np.shape(hrusp))
# print(np.shape(hrus))
# print(np.unique(hrus))

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot([1,2], [3,4])
# plt.show()

# os.listdir(tfile)
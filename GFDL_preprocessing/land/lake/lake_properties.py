import netCDF4 as nc
import numpy as np
import geospatialtools.gdal_tools as gdal_tools
lake_depth_max = 1.e10
lake_depth_min = 1.99
large_lake_sill_width = 200.0
max_plain_slope = -1.e10
dat_w_sat             = 1.000
dat_awc_lm2           = 1.000  
dat_k_sat_ref         = 0.021  
dat_psi_sat_ref       = -.059  
dat_chb               = 3.5  
dat_heat_capacity_ref = 0.0#8.4e7  
dat_thermal_cond_ref  = 8.4e7
dat_emis_dry          = 1.0#0.950  
dat_emis_sat          = 1.0#0.980  
dat_z0_momentum       = 1.4e-4 
dat_z0_momentum_ice   = 1.4e-4  
dat_tf_depr           = 0.00   
dat_refl_dry_dif = [0.060, 0.060]
dat_refl_dry_dir = [0.060, 0.060]
dat_refl_sat_dir = [0.060, 0.060]
dat_refl_sat_dif = [0.060, 0.060]

def Extract_Lake_Properties(cdir,ofile,lfrac,metadata):

 #Get the region metadata
 md = gdal_tools.retrieve_metadata('%s/mask_latlon.tif' % cdir)
 lat = (md['miny'] + md['maxy'])/2
 lon = (md['minx'] + md['maxx'])/2

 #Get the information from the original database
 fp = nc.Dataset(ofile)
 #lons = fp.variables['grid_x'][:]
 #lats = fp.variables['grid_y'][:]
 lons = fp.variables['x'][:]
 lats = fp.variables['y'][:]

 #Change to -180 to 180 if necessary
 lons[lons > 180] = lons[lons > 180] - 360

 #Find the match
 tmp = ((lons-lon)**2 + (lats-lat)**2)**0.5
 idx = np.where(tmp == np.min(tmp))
 ilat = idx[0][0]
 ilon = idx[1][0]
 #ilat = np.argmin(np.abs(lats-lat))
 #ilon = np.argmin(np.abs(lons-lon))
 #print ilat,ilon
 
 #Assign the properties
 output = {}
 #frac = np.ma.getdata(fp.variables['lake_frac'][ilat,ilon])
 #if frac < 0: frac = 0.0
 #if lfrac < 10**-3: lfrac = 0.0
 #if metadata['frac_override'] == True:
 # output['frac'] = np.array([np.ma.getdata(fp.variables['lake_frac'][ilat,ilon]),])
 #else:
 output['frac'] = np.array([lfrac,])
 #output['frac'] = np.array([0.0,])
 #output['frac'] = np.array([lfrac,])
 connected_to_next = np.ma.getdata(fp.variables['connected_to_next'][ilat,ilon])
 output['connected_to_next'] = connected_to_next
 output['whole_lake_area'] = np.ma.getdata(fp.variables['whole_lake_area'][ilat,ilon])
 #lake depth sill
 tmp = np.ma.getdata(fp.variables['lake_depth_sill'][ilat,ilon])
 tmp = min(tmp,lake_depth_max)
 tmp = max(tmp,lake_depth_min)
 output['lake_depth_sill'] = tmp
 #lake width sill
 lake_tau = np.ma.getdata(fp.variables['lake_tau'][ilat,ilon])
 #print connected_to_next,lake_tau
 if ((connected_to_next > 0.5) & (lake_tau > 1)):
  lake_width_sill = large_lake_sill_width
 else:
  lake_width_sill = -1.0
 output['lake_width_sill'] = lake_width_sill
 #backwater
 max_slope_to_next = np.ma.getdata(fp.variables['max_slope_to_next'][ilat,ilon])
 travel = np.ma.getdata(fp.variables['travel'][ilat,ilon])
 if ((travel < max_plain_slope) & (travel > 1.5)): lake_backwater = 1.0
 else: lake_backwater = 0.0
 if ((travel < max_plain_slope) & (travel < 1.5)): lake_backwater_1 = 1.0
 else: lake_backwater_1 = 0.0
 output['lake_backwater'] = lake_backwater
 output['lake_backwater_1'] = lake_backwater_1
 #Assign the soil variables
 output['w_sat'] = dat_w_sat
 output['awc_lm2'] = dat_awc_lm2
 output['k_sat_ref'] = dat_k_sat_ref
 output['psi_sat_ref'] = dat_psi_sat_ref
 output['chb'] = dat_chb
 output['alpha'] = 1.0
 output['heat_capacity_ref'] = dat_heat_capacity_ref
 output['thermal_cond_ref'] = dat_thermal_cond_ref
 output['refl_dry_dir'] = dat_refl_dry_dir
 output['refl_dry_dif'] = dat_refl_dry_dif
 output['refl_sat_dir'] = dat_refl_sat_dir
 output['refl_sat_dif'] = dat_refl_sat_dif
 output['emis_dry'] = dat_emis_dry
 output['emis_sat'] = dat_emis_sat
 output['z0_momentum'] = dat_z0_momentum
 output['z0_momentum_ice'] = dat_z0_momentum_ice
 output['ntile'] = 1
 #Assign the reservoir variables
 if ('reservoir' in metadata):
  if (metadata['reservoir'] == True):
   output['nrsv'] = np.ma.getdata(fp.variables['nrsv'][ilat,ilon])
   output['rsv_area'] = np.ma.getdata(fp.variables['rsv_area'][ilat,ilon])
   output['rsv_depth'] = np.ma.getdata(fp.variables['rsv_depth'][ilat,ilon])
   output['rsv_cap'] = np.ma.getdata(fp.variables['rsv_cap'][ilat,ilon])

 #Produce the map of lakes (HACK)
 undef = -9999.0
 metadata = gdal_tools.retrieve_metadata('%s/lakes_latlon.tif' % cdir)
 metadata['nodata'] = undef
 data = gdal_tools.read_raster('%s/lakes_latlon.tif' % cdir)
 data[data != undef] = 1
 gdal_tools.write_raster('%s/lake_tiles_latlon.tif' % cdir,metadata,data)

 # EZDEV: compute topographic 3d parameters for lake tile
 # read mask and topographic variables in EA masks
 lakes_mask = gdal_tools.read_raster('%s/lakes_ea.tif' % (cdir,))
 glaci_mask = gdal_tools.read_raster('%s/glaciers_ea.tif' % (cdir,))

 #EZDEV2 - add eamap of tiles:
#  lakes_tiles = gdal_tools.read_raster('%s/lakes_ea.tif' % (cdir,))
#  lakes_metadata = gdal_tools.retrieve_metadata('%s/lakes_ea.tif' % (cdir,))
#  lakes_tiles[lakes_tiles != undef] = 1
#  gdal_tools.write_raster('%s/glacier_tiles_ea.tif' % cdir,lakes_metadata,lakes_tiles)
#  print('----------------------------------------------------------')
#  print('EZDEV LAKES')
#  print('mask shape = ', lakes_mask.shape)
#  print('# of lakes cells = ', np.size(lakes_mask[lakes_mask > 0]))

 lak_and_gla_mask = np.logical_or(lakes_mask>0, glaci_mask>0)
 ncells_tot = np.size(lakes_mask)
 ncells_lak = np.size(lakes_mask[lakes_mask > 0])
 ncells_gla = np.size(glaci_mask[glaci_mask > 0])
 ncells_lak_and_gla = np.size(lak_and_gla_mask[lak_and_gla_mask > 0])

#  print('# of lake        cells = ', ncells_lak )
#  print('# of glacier     cells = ', ncells_gla )
#  print('# of lake + glac cells = ', ncells_lak_and_gla )
#  print('lake frac = {}'.format(ncells_lak/ncells_tot))
#  print('glac frac = {}'.format(ncells_gla/ncells_tot))
#  print('soil frac = {}'.format( 1-ncells_lak_and_gla/ncells_tot))


 svf_eamap = gdal_tools.read_raster('%s/svf_ea.tif' % (cdir,))
 tcf_eamap = gdal_tools.read_raster('%s/tcf_ea.tif' % (cdir,))
 sinssina_eamap = gdal_tools.read_raster('%s/sinssina_ea.tif' % (cdir,))
 sinscosa_eamap = gdal_tools.read_raster('%s/sinscosa_ea.tif' % (cdir,))
 radstelev_eamap = gdal_tools.read_raster('%s/radstelev_ea.tif' % (cdir,))
 radavelev_eamap = gdal_tools.read_raster('%s/radavelev_ea.tif' % (cdir,))
 elevation_eamap = gdal_tools.read_raster('%s/demns_ea.tif' % (cdir,))

#  print('svf map shape = ', np.shape(svf_eamap))
#  print('svf map mean = ', np.mean(svf_eamap))
#  print('elevation map shape = ', np.shape(elevation_eamap))
#  print('elevation map mean = ', np.mean(elevation_eamap))
# #  print(svf_eamap)

#  print('mask map shape = ', np.shape(lakes_mask))
#  print('mask uniques = ', np.unique(lakes_mask))

 svf_masked_lake = svf_eamap[lakes_mask > 0]
 tcf_masked_lake = tcf_eamap[lakes_mask > 0]
#  svf_masked_glacier = svf_eamap[glaci_mask > 0]
#  tcf_masked_glacier = tcf_eamap[glaci_mask > 0]
#  print('average svf lake = ', np.mean(svf_masked_lake))
#  print('average tcf lake = ', np.mean(tcf_masked_lake))
#  print('average svf glacier = ', np.mean(svf_masked_glacier))
#  print('average tcf glacier = ', np.mean(tcf_masked_glacier))
#  print('----------------------------------------------------------')

 sinscosa_masked_lake = sinscosa_eamap[lakes_mask > 0]
 sinssina_masked_lake = sinssina_eamap[lakes_mask > 0]
 radstelev_masked_lake = radstelev_eamap[lakes_mask > 0]
 radavelev_masked_lake = radavelev_eamap[lakes_mask > 0]
 elevation_masked_lake = elevation_eamap[lakes_mask > 0]

#  print('average elevation lake = ', np.mean(elevation_masked_lake))
#  print('min elevation lake = ', np.min(elevation_masked_lake))

#  output['svf'] = np.mean(svf_masked_lake)
#  output['tcf'] = np.mean(tcf_masked_lake)
#  output['sinscosa'] = np.mean(sinscosa_masked_lake)
#  output['sinssina'] = np.mean(sinssina_masked_lake)
 output['svf'] = np.mean(svf_masked_lake[svf_masked_lake > -900])
 output['tcf'] = np.mean(tcf_masked_lake[svf_masked_lake > -900])
 output['sinscosa'] = np.mean(sinscosa_masked_lake[sinscosa_masked_lake > -900])
 output['sinssina'] = np.mean(sinssina_masked_lake[sinssina_masked_lake > -900])
 output['radstelev'] = np.mean(radstelev_masked_lake[radstelev_masked_lake > -900])
 output['radavelev'] = np.mean(radavelev_masked_lake[radavelev_masked_lake > -900])
 output['elevation'] = np.mean(elevation_masked_lake[elevation_masked_lake > -900])


 # EZDEV END

 return output

def Extract_Lake_Properties_Old(lat,lon,file,metadata,ga):

 #Read in the soil type
 fp = nc.Dataset(file)
 lons = fp.variables['grid_x'][:]
 lats = fp.variables['grid_y'][:]
 frac = fp.variables['lake_frac'][:]

 #Change to -180 to 180 if necessary
 lons[lons > 180] = lons[lons > 180] - 360

 #Find the match
 ilat = np.argmin(np.abs(lats-lat))
 ilon = np.argmin(np.abs(lons-lon))
 frac = frac[ilat,ilon]

 #Assign the land cover information information
 output = {}
 frac = np.ma.getdata(fp.variables['lake_frac'][ilat,ilon])
 #if frac < 0: frac = 0.0
 #if frac < 10**-3: frac = 0.0
 #output['frac'] = np.ma.getdata(fp.variables['lake_frac'][ilat,ilon])
 output['frac'] = np.array([frac,])
 connected_to_next = np.ma.getdata(fp.variables['connected_to_next'][ilat,ilon])
 output['connected_to_next'] = connected_to_next
 output['whole_lake_area'] = np.ma.getdata(fp.variables['whole_lake_area'][ilat,ilon])
 #lake depth sill
 tmp = np.ma.getdata(fp.variables['lake_depth_sill'][ilat,ilon])
 tmp = min(tmp,lake_depth_max)
 tmp = max(tmp,lake_depth_min)
 output['lake_depth_sill'] = tmp
 #lake width sill
 lake_tau = np.ma.getdata(fp.variables['lake_tau'][ilat,ilon])
 #print connected_to_next,lake_tau
 if ((connected_to_next > 0.5) & (lake_tau > 1)):
  lake_width_sill = large_lake_sill_width
 else:
  lake_width_sill = -1.0
 output['lake_width_sill'] = lake_width_sill
 #backwater
 max_slope_to_next = np.ma.getdata(fp.variables['max_slope_to_next'][ilat,ilon])
 travel = np.ma.getdata(fp.variables['travel'][ilat,ilon])
 if ((travel < max_plain_slope) & (travel > 1.5)): lake_backwater = 1.0
 else: lake_backwater = 0.0
 if ((travel < max_plain_slope) & (travel < 1.5)): lake_backwater_1 = 1.0
 else: lake_backwater_1 = 0.0
 output['lake_backwater'] = lake_backwater
 output['lake_backwater_1'] = lake_backwater_1
 #Assign the soil variables
 output['w_sat'] = dat_w_sat
 output['awc_lm2'] = dat_awc_lm2
 output['k_sat_ref'] = dat_k_sat_ref
 output['psi_sat_ref'] = dat_psi_sat_ref
 output['chb'] = dat_chb
 output['alpha'] = 1.0
 output['heat_capacity_ref'] = dat_heat_capacity_ref
 output['thermal_cond_ref'] = dat_thermal_cond_ref
 output['refl_dry_dir'] = dat_refl_dry_dir
 output['refl_dry_dif'] = dat_refl_dry_dif
 output['refl_sat_dir'] = dat_refl_sat_dir
 output['refl_sat_dif'] = dat_refl_sat_dif
 output['emis_dry'] = dat_emis_dry
 output['emis_sat'] = dat_emis_sat
 output['z0_momentum'] = dat_z0_momentum
 output['z0_momentum_ice'] = dat_z0_momentum_ice
 output['nlake'] = 1

 return output

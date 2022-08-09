import netCDF4 as nc
import numpy as np
import geospatialtools.gdal_tools as gdal_tools
dat_w_sat             = 1.000
dat_awc_lm2           = 1.000  
dat_k_sat_ref         = 0.021  
dat_psi_sat_ref       = -.059  
dat_chb               = 3.5  
dat_heat_capacity_ref = 1.6e6
dat_thermal_cond_ref  = 1.8
dat_emis_dry          = 1.0#0.950  
dat_emis_sat          = 1.0#0.980  
dat_z0_momentum       = 0.01
dat_tf_depr           = 0.00   
dat_refl_max_dif = [0.80,0.80]
dat_refl_max_dir = [0.80,0.80]
dat_refl_min_dir = [0.650,0.650]
dat_refl_min_dif = [0.650,0.650]

def Extract_Glacier_Properties(cdir,gfrac):

 #Assign the land cover information information
 output = {}
 output['frac'] = np.array([gfrac,])#frac[8]
 #Assign the soil variables
 output['w_sat'] = dat_w_sat
 output['awc_lm2'] = dat_awc_lm2
 output['k_sat_ref'] = dat_k_sat_ref
 output['psi_sat_ref'] = dat_psi_sat_ref
 output['chb'] = dat_chb
 output['alpha'] = 1.0
 output['heat_capacity_ref'] = dat_heat_capacity_ref
 output['thermal_cond_ref'] = dat_thermal_cond_ref
 output['refl_max_dir'] = dat_refl_max_dir
 output['refl_max_dif'] = dat_refl_max_dif
 output['refl_min_dir'] = dat_refl_min_dir
 output['refl_min_dif'] = dat_refl_min_dif
 output['emis_dry'] = dat_emis_dry
 output['emis_sat'] = dat_emis_sat
 output['z0_momentum'] = dat_z0_momentum
 tfreeze = 273.16
 output['tfreeze'] = tfreeze - dat_tf_depr
 output['ntile'] = 1

 #Produce the map of glaciers (HACK)
 undef = -9999.0
 metadata = gdal_tools.retrieve_metadata('%s/glaciers_latlon.tif' % cdir)
 metadata['nodata'] = undef
 data = gdal_tools.read_raster('%s/glaciers_latlon.tif' % cdir)
 data[data != undef] = 1
 gdal_tools.write_raster('%s/glacier_tiles_latlon.tif' % cdir,metadata,data)

  # EZDEV: compute topographic 3d parameters for glacier tile
 # read mask and topographic variables in EA masks
#  lakes_mask = gdal_tools.read_raster('%s/lakes_ea.tif' % (cdir,))
 glaci_mask = gdal_tools.read_raster('%s/glaciers_ea.tif' % (cdir,))
 #EZDEV2 - add eamap of tiles:
#  glaci_tiles = gdal_tools.read_raster('%s/glaciers_ea.tif' % (cdir,))
#  glaci_metadata = gdal_tools.retrieve_metadata('%s/glaciers_ea.tif' % (cdir,))
#  glaci_tiles[glaci_tiles != undef] = 1
#  gdal_tools.write_raster('%s/glacier_tiles_ea.tif' % cdir,glaci_metadata,glaci_tiles)
#  print('----------------------------------------------------------')
#  print('EZDEV GLACIER TOPO3D')

 svf_eamap = gdal_tools.read_raster('%s/svf_ea.tif' % (cdir,))
 tcf_eamap = gdal_tools.read_raster('%s/tcf_ea.tif' % (cdir,))
 sinssina_eamap = gdal_tools.read_raster('%s/sinssina_ea.tif' % (cdir,))
 sinscosa_eamap = gdal_tools.read_raster('%s/sinscosa_ea.tif' % (cdir,))
 radstelev_eamap = gdal_tools.read_raster('%s/radstelev_ea.tif' % (cdir,))
 radavelev_eamap = gdal_tools.read_raster('%s/radavelev_ea.tif' % (cdir,))
 elevation_eamap = gdal_tools.read_raster('%s/demns_ea.tif' % (cdir,))

 svf_masked_glacier = svf_eamap[glaci_mask > 0]
 tcf_masked_glacier = tcf_eamap[glaci_mask > 0]
 sinscosa_masked_glacier = sinscosa_eamap[glaci_mask > 0]
 sinssina_masked_glacier = sinssina_eamap[glaci_mask > 0]
 radstelev_masked_glacier = radstelev_eamap[glaci_mask > 0]
 radavelev_masked_glacier = radavelev_eamap[glaci_mask > 0]
 elevation_masked_glacier = elevation_eamap[glaci_mask > 0]

 output['svf'] = np.mean(svf_masked_glacier[svf_masked_glacier > -900])
 output['tcf'] = np.mean(tcf_masked_glacier[tcf_masked_glacier > -900])
 output['sinscosa'] = np.mean(sinscosa_masked_glacier[sinscosa_masked_glacier > -900])
 output['sinssina'] = np.mean(sinssina_masked_glacier[sinssina_masked_glacier > -900])
 output['radstelev'] = np.mean(radstelev_masked_glacier[radstelev_masked_glacier > -900])
 output['radavelev'] = np.mean(radavelev_masked_glacier[radavelev_masked_glacier > -900])
 output['elevation'] = np.mean(elevation_masked_glacier[elevation_masked_glacier > -900])
 # END TOPO3D

 return output

def Extract_Glacier_Properties_Old(metadata,file,ga):

 minlat = metadata['minlat']
 maxlat = metadata['maxlat']
 minlon = metadata['minlon']
 maxlon = metadata['maxlon']
 #open access to the file
 ga("sdfopen %s" % file)
 ga("set gxout geotiff")
 #ga("set z 9")
 ga("set z 1 10")
 ga("value = aave(frac,lon=%f,lon=%f,lat=%f,lat=%f)" % (minlon,maxlon,minlat,maxlat))
 ga("set x 1")
 ga("set y 1")
 ga("set t 1")
 frac = ga.expr("value")[8]
 ga("close 1")

 #Assign the land cover information information
 output = {}
 output['frac'] = np.array([frac,])#frac[8]
 #Assign the soil variables
 output['w_sat'] = dat_w_sat
 output['awc_lm2'] = dat_awc_lm2
 output['k_sat_ref'] = dat_k_sat_ref
 output['psi_sat_ref'] = dat_psi_sat_ref
 output['chb'] = dat_chb
 output['alpha'] = 1.0
 output['heat_capacity_ref'] = dat_heat_capacity_ref
 output['thermal_cond_ref'] = dat_thermal_cond_ref
 output['refl_max_dir'] = dat_refl_max_dir
 output['refl_max_dif'] = dat_refl_max_dif
 output['refl_min_dir'] = dat_refl_min_dir
 output['refl_min_dif'] = dat_refl_min_dif
 output['emis_dry'] = dat_emis_dry
 output['emis_sat'] = dat_emis_sat
 output['z0_momentum'] = dat_z0_momentum
 tfreeze = 273.16
 output['tfreeze'] = tfreeze - dat_tf_depr
 output['nglacier'] = 1

 return output

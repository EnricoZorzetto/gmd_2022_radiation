import pickle
import os
import sys

def create_metadata(region):

 if region == 'C384.v20150402_k1dh10p1':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':3600,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Nathaniel.Chaney/grids/C384.v20150402/C384.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Nathaniel.Chaney/grids/C384.v20150402/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/c384_025/hydrography.c384-v20141201_025-v20140610_20141204.tile$tid.nc',
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':1,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':1,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'min_nbands':3, #min number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'c96_OM4_025_grid_No_mg_drag_v20160808_k1dh10p1':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':720,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/C96_grid.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/c96_OM4_025_hydrography_v20170413/river_data.tile$tid.nc',
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':1,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':1,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'min_nbands':3, #min number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'c96_OM4_05_uni_grid.v20181218_k2dh10p1':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':720,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f2/dev/Nathaniel.Chaney/data/grids/c96_OM4_05_uni_grid.v20181218/C96_grid.tile$tid.nc',
            'lm_template':'/lustre/f2/dev/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f2/dev/Nathaniel.Chaney/data/hydrography/river_data_hydrography.c96_OM4_05-v20151028_20161005/river_data.tile$tid.nc',
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':1,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'min_nbands':3, #min number of height bands
                         'channel_threshold':10**5,
                         'hcov':['dem',],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'mosaic_c96.v20180227_k2dh10p1':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':720,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f2/dev/Nathaniel.Chaney/data/grids/mosaic_c96.v20180227/C96_grid.tile$tid.nc',
            'lm_template':'/lustre/f2/dev/Nathaniel.Chaney/data/grids/mosaic_c96.v20180227/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f2/dev/Nathaniel.Chaney/data/hydrography/river_data_hydrography.c96_OM4_05-v20151028_20161005/river_data.tile$tid.nc',
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':1,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'min_nbands':3, #min number of height bands
                         'channel_threshold':10**5,
                         'hcov':['dem',],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'c96_OM4_025_grid_No_mg_drag_v20160808_DTB_simple':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':720,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/C96_grid.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/c96_OM4_025_hydrography_v20170413/river_data.tile$tid.nc',
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':1,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':1,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'min_nbands':3, #min number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'chaney2017_global_1deg':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':-90.0,
            'maxlat':90.0,
            'minlon':0.0,
            'maxlon':360.0,
            'res':1.0,
            'npes':720,
            'fsres':3.0/3600.0, #arcdegrees
            'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            #'meteorology':{
            #            'iyear':2002,
            #            'fyear':2014,
            #            },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/1deg_cm2/hydrography.1deg_cm2.20131203.tile1.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':5,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':2,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'c96_OM4_025_grid_No_mg_drag_v20160808_DTB':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':720,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/C96_grid.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/c96_OM4_025_hydrography_v20170413/river_data.tile$tid.nc',
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':3,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':10, #max number of height bands
                         'min_nbands':3, #min number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'chaney2017_global_dev':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':37.0,#44.0,#40.0,#37.0,#31.0,#40.0,#31.0,#22.0,#31.0,
            'maxlat':38.0,#45.0,#41.0,#38.0,#32.0,#41.0,#32.0,#23.0,#32.0,
            'minlon':-81.0,#-108.0,#-80.0,#-120.0,#-92.0,#-80.0,#104.0,#-92.0,
            'maxlon':-80.0,#-107.0,#-79.0,#-119.0,#-91.0,#-79.0,#105.0,#-91.0,
            'res':1.0,
            'npes':1,
            'fsres':3.0/3600.0, #arcdegrees
            'fsres_meters':90.0, #meters
            'dir':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            'ntiles':1,
            'river_network':'/lustre/f2/dev/Krista.A.Dunne/mod_input_files/1deg_cm2/hydrography.1deg_cm2.20131203.tile1.nc',
            'gs_template':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':1,#number of clusters per height band
                         'max_nbands':5, #max number of height bands
                         'min_nbands':2, #min number of height bands
                         'channel_threshold':10**5,
                         #'hcov':['slope','tas','prec','width_slope','relief_a',
                         #         'relief_b','length','relief'],
                         'hcov':['dem',],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'chaney2017_global_dev_mar2021':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':36.0,#44.0,#40.0,#37.0,#31.0,#40.0,#31.0,#22.0,#31.0,
            'maxlat':41.0,#45.0,#41.0,#38.0,#32.0,#41.0,#32.0,#23.0,#32.0,
            'minlon':-95.0,#-108.0,#-80.0,#-120.0,#-92.0,#-80.0,#104.0,#-92.0,
            'maxlon':-85.0,#-107.0,#-79.0,#-119.0,#-91.0,#-79.0,#105.0,#-91.0,
            'res':1.0,
            'npes':32,
            'fsres':3.0/3600.0, #arcdegrees
            'fsres_meters':90.0, #meters
            'dir':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            'ntiles':1,
            'river_network':'/lustre/f2/dev/Krista.A.Dunne/mod_input_files/1deg_cm2/hydrography.1deg_cm2.20131203.tile1.nc',
            'gs_template':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f2/dev/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':1,#number of clusters per height band
                         'max_nbands':5, #max number of height bands
                         'min_nbands':2, #min number of height bands
                         'channel_threshold':10**5,
                         #'hcov':['slope','tas','prec','width_slope','relief_a',
                         #         'relief_b','length','relief'],
                         'hcov':['dem',],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'C384.v20150402_k2dh25p3':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':3200,
            'fsres':3.0/3600.0, #arcdegrees
            'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Nathaniel.Chaney/grids/C384.v20150402/C384.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Nathaniel.Chaney/grids/C384.v20150402/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/c384_025/hydrography.c384-v20141201_025-v20140610_20141204.tile$tid.nc',
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }

 if region == 'c96_OM4_025_grid_No_mg_drag_v20160808':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':32,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/C96_grid.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/grids/c96_OM4_025_grid_No_mg_drag_v20160808/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/c96_OM4_025_hydrography_v20170413/river_data.tile$tid.nc',
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':5,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':2,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'CONUS0.25deg_9IRR_090617':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':24.0,
            'maxlat':50.0,
            'minlon':-126.0,
            'maxlon':-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':1,#640,
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'land_fractions':'original',
            'political_boundaries':'CONUS',
            'river_network':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170406.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'reservoir':True,
            'topography':{
                      'type':'ned',
                      },
            'soil':{
                      'type':'polaris',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cld',
                   },
            'irrigation':{
                      'type':'default',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':1,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':3, #max number of height bands
                         'channel_threshold':10**4,
                         'hcov':['slope','tas','prec','width_slope','slope_slope',
                                  'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                },
                       }
            }
 if region == 'chaney2017_global_v4':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':-90.0,
            'maxlat':90.0,
            'minlon':0.0,
            'maxlon':360.0,
            'res':1.0,
            'npes':720,
            'fsres':3.0/3600.0, #arcdegrees
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            #'meteorology':{
            #            'iyear':2002,
            #            'fyear':2014,
            #            },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/1deg_cm2/hydrography.1deg_cm2.20131203.tile1.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':5,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':2,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'chaney2017_global_v3':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':-60.0,
            'maxlat':60.0,
            'minlon':-140.0,
            'maxlon':-30.0,
            'res':1.0,
            'npes':720,
            'fsres':3.0/3600.0, #arcdegrees
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            #'meteorology':{
            #            'iyear':2002,
            #            'fyear':2014,
            #            },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/1deg_cm2/hydrography.1deg_cm2.20131203.tile1.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':5,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':2,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'chaney2017_global_v2':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':20.0,
            'maxlat':60.0,
            'minlon':-140.0,
            'maxlon':-50.0,
            'res':1.0,
            'npes':360,
            'fsres':3.0/3600.0, #arcdegrees
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            #'meteorology':{
            #            'iyear':2002,
            #            'fyear':2014,
            #            },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/1deg_cm2/hydrography.1deg_cm2.20131203.tile1.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':5,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':2,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'chaney2017_global':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':42.25,#36.0,
            'maxlat':42.50,#44.0,
            'minlon':-80.75,#-121.0,
            'maxlon':-80.5,#-117.0,
            'res':0.25,
            'npes':1,#36,
            'fsres':3.0/3600.0, #arcdegrees
            'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            #'meteorology':{
            #            'iyear':2002,
            #            'fyear':2014,
            #            },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170202.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':5,#number of hillslopes
                         'dh':10,#elevation difference between adjacent height bands
                         'p':2,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'dev_irr':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':48.25,
            'maxlat':48.50,
            'minlon':-95.50,
            'maxlon':-95.25,
            'res':0.25,
            'npes':1,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            'meteorology':{
                        'iyear':2002,
                        'fyear':2014,
                        },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170202.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'ned',
                      },
            'soil':{
                      'type':'polaris',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cld',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':1,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':3, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0},
                                 'c32c4':{'min':0.0,'max':1.0},
                                 #'cheight':{'min':0.0,'max':50.0},
                                },
                       }
            }
 if region == 'dev_chaney2017':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':36.0,
            'maxlat':36.250,
            'minlon':-118.90,
            'maxlon':-118.65,
            'res':0.25,
            'npes':1,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            'meteorology':{
                        'iyear':2002,
                        'fyear':2014,
                        },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170202.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'ned',
                      },
            'soil':{
                      'type':'polaris',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cld',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':5, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0},
                                 #'c32c4':{'min':0.0,'max':1.0},
                                 #'cheight':{'min':0.0,'max':50.0},
                                },
                       }
            }
 if region == 'chaney2017':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':36.0,
            'maxlat':36.250,
            'minlon':-118.90,
            'maxlon':-118.65,
            'res':0.25,
            'npes':1,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            'meteorology':{
                        'iyear':2002,
                        'fyear':2014,
                        },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170202.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'ned',
                      },
            'soil':{
                      'type':'polaris',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cld',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 #'c32c4':{'min':0.0,'max':1.0},
                                 #'cheight':{'min':0.0,'max':50.0},
                                },
                       }
            }
 if region == 'chaney2017_expanded':

  metadata = {
            'name':region,
            'grid':'new',
            'minlat':35.750,
            'maxlat':36.50,
            'minlon':-119.15,
            'maxlon':-118.40,
            'res':0.25,
            'npes':1,
            'fsres':1.0/3600.0, #arcdegrees
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'political_boundaries':'undefined',
            'land_fractions':'updated',
            'meteorology':{
                        'type':'cutout',
                        'iyear':2002,
                        'fyear':2014,
                        },
            'ntiles':1,
            'river_network':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170202.nc',
            'gs_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/grid.tile$tid.nc' % region,
            'lm_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/grid_spec/land_mask.tile$tid.nc' % region,
            'rn_template':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s/river/hydrography.tile$tid.nc' % region,
            'topography':{
                      'type':'srtm',
                      },
            'soil':{
                      'type':'polaris',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cld',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**5,
                         'hcov':['slope','tas','prec','width_slope','relief_a',
                                 'relief_b','length','relief'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 #'c32c4':{'min':0.0,'max':1.0},
                                 #'cheight':{'min':0.0,'max':50.0},
                                },
                       }
            }
 if region == 'c96_OM4_025_uni_grid.v20150522_Original':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':640,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Sergey.Malyshev/DATA/grids/c96_OM4_025_uni_grid.v20150522/C96_grid.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Sergey.Malyshev/DATA/grids/c96_OM4_025_uni_grid.v20150522/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/c96_cm4/hydrography.c96_cm4.20140617.tile$tid.nc',
            'topography':{
                      'type':'gmted2010',
                      },
            'soil':{
                      'type':'original',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                      'type':'original',
                      'NN':10,
                   }
            }
 if region == 'c192_OM4_025_grid_No_mg_drag_v20160808':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':3200,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Nathaniel.Chaney/grids/c192_OM4_025_grid_No_mg_drag_v20160808/C192_grid.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Nathaniel.Chaney/grids/c192_OM4_025_grid_No_mg_drag_v20160808/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Nathaniel.Chaney/hydrography/c192_OM4_025_hydrography_v20170413/river_data.tile$tid.nc',
            'topography':{
                      'type':'gmted2010',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':10, #max number of height bands
                         'channel_threshold':10**6,
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'hcov':['slope','tas','prec','width_slope','slope_slope',
                                  'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                },
                       }
            }
 if region == 'c96_OM4_025_uni_grid.v20150522':

  metadata = {
            'name':region,
            'grid':'predefined',
            'npes':640,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'ntiles':6,
            'gs_template':'/lustre/f1/unswept/Sergey.Malyshev/DATA/grids/c96_OM4_025_uni_grid.v20150522/C96_grid.tile$tid.nc',
            'lm_template':'/lustre/f1/unswept/Sergey.Malyshev/DATA/grids/c96_OM4_025_uni_grid.v20150522/land_mask_tile$tid.nc',
            'rn_template':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/c96_cm4/hydrography.c96_cm4.20140617.tile$tid.nc',
            'topography':{
                      'type':'gmted2010',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':20, #max number of height bands
                         'channel_threshold':10**6,
                         'hcov':['slope','tas','prec','width_slope','slope_slope',
                                  'length','relief'],
                         #'hcov':['slope','tas','prec','width_slope','slope_slope',
                         #         'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                 'c32c4':{'min':0.0,'max':1.0},
                                },
                       }
            }
 if region == 'DEV_GLOBE':

  metadata = {
            #-90,90,-20,60
            'minlat':-90,#0,#-56,#42.0,#24.0,
            'maxlat':90,#1,#12,#43.0,#50.0,
            'minlon':0,#-80,#-82,#-125.0,#-126.0,
            'maxlon':60,#-79,#-34,#-124.0,#-65.0,
	    'res':1.0,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'ntiles':1,
            'npes':640,
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'river_network':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/1deg_cm2/hydrography.1deg_cm2.20131203.tile1.nc',
            'topography':{
                      'type':'gmted2010',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'globcover2009',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':10, #max number of height bands
                         'channel_threshold':10**6,
                         'hcov':['slope','tas','prec','width_slope','slope_slope',
                                  'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                },
                       }
            }
 if region == 'GLOBE_1DEG':

  metadata = {
            #-90,90,-20,60
            'minlat':-90,#0,#-56,#42.0,#24.0,
            'maxlat':90,#1,#12,#43.0,#50.0,
            'minlon':0,#-80,#-82,#-125.0,#-126.0,
            'maxlon':360,#-79,#-34,#-124.0,#-65.0,
	    'res':1.0,
            'fsres':3.0/3600.0, #arcdegrees
	    'fsres_meters':100, #meters
            'ntiles':1,
            'npes':640,
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'land_fractions':'original',
            'political_boundaries':'undefined',
            'river_network':'/lustre/f1/unswept/Krista.A.Dunne/mod_input_files/1deg_cm2/hydrography.1deg_cm2.20131203.tile1.nc',
            'topography':{
                      'type':'gmted2010',
                      },
            'soil':{
                      'type':'soilgrids',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cci',
                   },
            'irrigation':{
                      'type':'undefined',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':2,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':10, #max number of height bands
                         'channel_threshold':10**6,
                         'hcov':['slope','tas','prec','width_slope','slope_slope',
                                  'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                },
                       }
            }
 elif region == 'CONUS0.25deg_9IRR_062817':

  metadata = {
            'grid':'new',
            'minlat':24.0,
            'maxlat':50.0,
            'minlon':-126.0,
            'maxlon':-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':640,
            'dir':'/lustre/f1/Nathaniel.Chaney/predefined_input/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'land_fractions':'original',
            'political_boundaries':'CONUS',
            'river_network':'/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170406.nc',
            'reservoir':True,
            'topography':{
                      'type':'ned',
                      },
            'soil':{
                      'type':'polaris',
                   },
            'geohydrology':{
                      'type':'soilgrids',
                   },
            'landcover':{
                      'type':'cld',
                   },
            'irrigation':{
                      'type':'default',
                   },
            'climate':{
                      'type':'worldclim',
                   },
            'lake':{
                      'type':'hydrolakes',
                   },
            'glacier':{
                      'type':'unknown',
                   },
            'hillslope':{
                         'type':'chaney2017',
                         'k':1,#number of hillslopes
                         'dh':25,#elevation difference between adjacent height bands
                         'p':3,#number of clusters per height band
                         'max_nbands':3, #max number of height bands
                         'channel_threshold':10**4,
                         'hcov':['slope','tas','prec','width_slope','slope_slope',
                                  'length','relief','x_aspect','y_aspect'],
                         'tcov':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                },
                       }
            }

 if region == 'DEV':

  nhillslope = 1
  ntiles = 10
  tile_length = 100.0
  tile_relief = 25.0
  max_ntiles = 1
  #Make metadata
  metadata = {
            'minlat':42.0,#24.0,
            'maxlat':44.0,#50.0,
            'minlon':-125.0,#-126.0,
            'maxlon':-124.0,#-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':32,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'frac_override':True,
            'fsoil':{
                    'type':'original',
                   },
            'fgeohydrology':{
                    'type':'original',
                   },
            'fhillslope':{
                    'type':'original',
                    'NN':10,
                   },
            'fvegetation':{
                    'type':'original',
                   },
            'ffractions':'original',
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },

			 #'hillslope':{'slope':{'t':0.1},
                         #             'latitude':{'t':1.0},
                         #             'longitude':{'t':1.0},
                         #             'tas':{'t':5},
                         #             'prec':{'t':100},
                         #             'rwidth':{'t':1.0},
                         #             'length':{'t':1.0}},
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                }
                         #'tile':{'maxsmc':{'t':0.1},
                         #        'c2n':{'t':0.25},
                         #        'g2t':{'t':0.25},
                         #        'd2e':{'t':0.25},
                         #        'n2u':{'t':0.25},
                         #        'ksat':{'t':0.1},
                         #        'irr':{'t':1.0}
                         #       }
                         },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }
 if region == 'CONUS_BASELINE':

  nhillslope = 1
  ntiles = 10
  tile_length = 100.0
  tile_relief = 25.0
  max_ntiles = 1
  #Make metadata
  metadata = {
            'minlat':24.0,
            'maxlat':50.0,
            'minlon':-126.0,
            'maxlon':-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':640,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'frac_override':True,
            'fsoil':{
                    'type':'original',
                   },
            'fgeohydrology':{
                    'type':'original',
                   },
            'fhillslope':{
                    'type':'original',
                    'NN':10,
                   },
            'fvegetation':{
                    'type':'original',
                   },
            'ffractions':'original',
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },

			 #'hillslope':{'slope':{'t':0.1},
                         #             'latitude':{'t':1.0},
                         #             'longitude':{'t':1.0},
                         #             'tas':{'t':5},
                         #             'prec':{'t':100},
                         #             'rwidth':{'t':1.0},
                         #             'length':{'t':1.0}},
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                }
                         #'tile':{'maxsmc':{'t':0.1},
                         #        'c2n':{'t':0.25},
                         #        'g2t':{'t':0.25},
                         #        'd2e':{'t':0.25},
                         #        'n2u':{'t':0.25},
                         #        'ksat':{'t':0.1},
                         #        'irr':{'t':1.0}
                         #       }
                         },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }
 if region == 'DEV_IRR_1TILE':

  nhillslope = 1
  ntiles = 1#3
  tile_length = 100.0
  tile_relief = 0.1
  max_ntiles = 1#3
  #Make metadata
  metadata = {
            'minlat':35.0,#39.250,#42.50,#40.750,#35.750,#35.0,#36.0,
            'maxlat':35.25,#39.50,#42.750,#41.0,#36.0,#35.25,#36.25,
            'minlon':-120.75,#-122.0,#-114.0,#-98.0,#-90.50,#-120.75,#-118.9,
            'maxlon':-120.50,#-121.750,#-113.750,#-97.750,#-90.25,#-120.50,#-118.65,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':1,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':False,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },

			 #'hillslope':{'slope':{'t':0.1},
                         #             'latitude':{'t':1.0},
                         #             'longitude':{'t':1.0},
                         #             'tas':{'t':5},
                         #             'prec':{'t':100},
                         #             'rwidth':{'t':1.0},
                         #             'length':{'t':1.0}},
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                }
                         #'tile':{'maxsmc':{'t':0.1},
                         #        'c2n':{'t':0.25},
                         #        'g2t':{'t':0.25},
                         #        'd2e':{'t':0.25},
                         #        'n2u':{'t':0.25},
                         #        'ksat':{'t':0.1},
                         #        'irr':{'t':1.0}
                         #       }
                         },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            'meteorology':{ 
                        'iyear':2002,
                        'fyear':2003,
                        'flag':False,
                        }
            }
 if region == 'DEV_IRR':

  nhillslope = 1
  ntiles = 3
  tile_length = 100.0
  tile_relief = 0.1
  max_ntiles = 3
  #Make metadata
  metadata = {
            'minlat':35.750,#35.0,#36.0,
            'maxlat':36.0,#35.25,#36.25,
            'minlon':-119.50,#-120.75,#-118.9,
            'maxlon':-119.250,#-120.50,#-118.65,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':1,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':False,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },

			 #'hillslope':{'slope':{'t':0.1},
                         #             'latitude':{'t':1.0},
                         #             'longitude':{'t':1.0},
                         #             'tas':{'t':5},
                         #             'prec':{'t':100},
                         #             'rwidth':{'t':1.0},
                         #             'length':{'t':1.0}},
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                }
                         #'tile':{'maxsmc':{'t':0.1},
                         #        'c2n':{'t':0.25},
                         #        'g2t':{'t':0.25},
                         #        'd2e':{'t':0.25},
                         #        'n2u':{'t':0.25},
                         #        'ksat':{'t':0.1},
                         #        'irr':{'t':1.0}
                         #       }
                         },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            'meteorology':{ 
                        'iyear':2002,
                        'fyear':2003,
                        'flag':False,
                        }
            }
 if region == 'DEV2':

  nhillslope = 10
  ntiles = 3
  tile_length = 100.0
  max_ntiles = 10
  #Make metadata
  metadata = {
            'minlat':36.0,
            'maxlat':38.0,
            'minlon':-120.0,
            'maxlon':-119.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':32,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':False,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'convergence':True,
			 'hillslope':{'slope':{'t':0.1},
                                      'latitude':{'t':0.1},
                                      'longitude':{'t':0.1},
                                      'elevation':{'t':100},
                                      'rwidth':{'t':1.0},
                                      'length':{'t':1.0}},
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile':{'maxsmc':{'t':0.1},
                                 'c2n':{'t':0.25},
                                 'g2t':{'t':0.25},
                                 'ksat':{'t':0.1},
                                 'irr':{'t':0.25}
                                }
                         },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }

 if region == 'DEV3':

  nhillslope = 5
  ntiles = 3
  tile_length = 100.0
  tile_relief = 25.0
  max_ntiles = 10
  #Make metadata
  metadata = {
            'minlat':36.0,#35.0,#36.0,
            'maxlat':36.25,#35.25,#36.25,
            'minlon':-118.9,#-120.75,#-118.9,
            'maxlon':-118.65,#-120.50,#-118.65,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':1,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':False,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },

			 #'hillslope':{'slope':{'t':0.1},
                         #             'latitude':{'t':1.0},
                         #             'longitude':{'t':1.0},
                         #             'tas':{'t':5},
                         #             'prec':{'t':100},
                         #             'rwidth':{'t':1.0},
                         #             'length':{'t':1.0}},
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 #'irr':{'min':0.0,'max':1.0}
                                }
                         #'tile':{'maxsmc':{'t':0.1},
                         #        'c2n':{'t':0.25},
                         #        'g2t':{'t':0.25},
                         #        'd2e':{'t':0.25},
                         #        'n2u':{'t':0.25},
                         #        'ksat':{'t':0.1},
                         #        'irr':{'t':1.0}
                         #       }
                         },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            'meteorology':{ 
                        'iyear':2002,
                        'fyear':2003,
                        'flag':True,
                        }
            }

 elif region == 'DEV_WARSAW':

  nhillslope = 1
  ntiles = 3
  tile_length = 100.0
  tile_relief = 10.0
  max_ntiles = 3
  #Make metadata
  metadata = {
            'minlat':36.50,#38.75,#24.0,
            'maxlat':37.0,#39.25,#50.0,
            'minlon':-93.50,#-120.50,#-126.0,
            'maxlon':-93.0,#-120.0,#-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':4,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':True,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                }
                        },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }
 elif region == 'CONUS0.25deg_9IRR_DEV_CELL':

  nhillslope = 1
  ntiles = 3
  tile_length = 100.0
  tile_relief = 10.0
  max_ntiles = 3
  #Make metadata
  metadata = {
            'minlat':38.75,#24.0,
            'maxlat':39.0,#50.0,
            'minlon':-120.25,#-126.0,
            'maxlon':-120.0,#-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':1,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':True,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                }
                        },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }
 elif region == 'CONUS0.25deg_9IRR_DEV':

  nhillslope = 1
  ntiles = 3
  tile_length = 100.0
  tile_relief = 10.0
  max_ntiles = 3
  #Make metadata
  metadata = {
            'minlat':34.0,#24.0,
            'maxlat':42.0,#50.0,
            'minlon':-123.0,#-126.0,
            'maxlon':-119.0,#-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':160,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':True,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                }
                        },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }

 elif region == 'CONUS0.25deg_9IRR':

  nhillslope = 1
  ntiles = 3
  tile_length = 100.0
  tile_relief = 10.0
  max_ntiles = 3
  #Make metadata
  metadata = {
            'minlat':24.0,
            'maxlat':50.0,
            'minlon':-126.0,
            'maxlon':-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':640,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':True,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                }
                        },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }

 elif region == 'CONUS0.25deg_k3dh25p2':

  nhillslope = 3
  ntiles = 2
  tile_length = 100.0
  tile_relief = 25.0
  max_ntiles = 20
  #Make metadata
  metadata = {
            'minlat':24.0,
            'maxlat':50.0,
            'minlon':-126.0,
            'maxlon':-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':640,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'meteo_tas':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_tmean_30yr_normal_800mM2_annual_bil.bil',
            'meteo_prec':'/lustre/f1/unswept/Nathaniel.Chaney/data/prism/800m/PRISM_ppt_30yr_normal_800mM2_annual_bil.bil',
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':True,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'hillslope_threshold':12,#percent
                         'tile_threshold':12,#percent
                         'convergence':False,
                         'hillslope':{'slope':{'min':0.0,'max':0.6},
                                      #'latitude':{'min':0.0,'max':1.0},
                                      #'longitude':{'min':0.0,'max':1.0},
                                      'tas':{'min':5.0,'max':20.0},
                                      'prec':{'min':300.0,'max':2000.0},
                                      'width_slope':{'min':-1,'max':1},
                                      'slope_slope':{'min':-1,'max':1},
                                      'length':{'min':100.0,'max':1000.0},
                                      'relief':{'min':0,'max':400},
                                      'x_aspect':{'min':-1,'max':1},
                                      'y_aspect':{'min':-1,'max':1}
                                      },
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile_relief':tile_relief,
                         'tile':{'maxsmc':{'min':0.0,'max':1.0},
                                 'c2n':{'min':0.0,'max':1.0},
                                 'g2t':{'min':0.0,'max':1.0},
                                 'd2e':{'min':0.0,'max':1.0},
                                 #'n2u':{'min':0.0,'max':1.0},
                                 'ksat':{'min':0.0,'max':100.0},
                                 'irr':{'min':0.0,'max':1.0}
                                }
                        },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }

 elif region == 'CONUS0.25deg_9tiles_0418':

  nhillslope = 1
  ntiles = 3
  tile_length = 100.0
  max_ntiles = 3
  #Make metadata
  metadata = {
            'minlat':24.0,
            'maxlat':50.0,
            'minlon':-126.0,
            'maxlon':-65.0,
	    'res':0.25,
            'fsres':1.0/3600.0, #arcdegrees
	    'fsres_meters':30, #meters
            'ntiles':1,
            'npes':640,
            'dem_fine':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'dem_coarse':'/lustre/f1/unswept/Nathaniel.Chaney/data/NED/NED.vrt',
            'soil_dataset':'/lustre/f1/unswept/Nathaniel.Chaney/data/soilgrids',
            'geohydrology':'/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/geohydrology',
            'landcover':'/lustre/f1/unswept/Nathaniel.Chaney/data/CDL/2016_30m_cdls.img',
            'baresoil':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/bare.vrt',
            'treecover':'/lustre/f1/unswept/Nathaniel.Chaney/data/USGS/treecover.vrt',
            'irrigation':'/lustre/f1/unswept/Nathaniel.Chaney/data/irrigation/mirad250_12v3_grid/mirad250_12v3',
            'meteomask':'/lustre/f1/unswept/Nathaniel.Chaney/data/pcf/0.25deg/annual/meteo_mask.tif', # CAREFUL!
            'dir':'/lustre/f1/Nathaniel.Chaney/AMS2017/%s' % region,
            'grid_template':'horizontal_grid.nc',
            'river_template':'river_data.nc',
            'gw':'tiled',
            'lake_override':True,
            'soil':{
                    'type':'original',
                   },
            'clustering':{
                         'convergence':False,
                         'hillslope':{'slope':{'t':0.1}},
                         'max_ntiles':max_ntiles,
                         'tile_length':tile_length,
                         'tile':{'maxsmc':{'t':0.1},
                                 'c2n':{'t':0.25},
                                 'g2t':{'t':0.25},
                                 'ksat':{'t':0.1},
                                 'irr':{'t':0.25}
                                }

                         },
            'hillslope':{  
                         'type':'original',
                         'nhillslope':nhillslope,
                         'ntiles':ntiles, #clustering per elevation tile
                         'tile_length':tile_length, #meters
                         'max_ntiles':max_ntiles,
                        },
            }

 return metadata

#Define parameters
#region = 'DEV'
region = sys.argv[1]

#Create the metadata
metadata = create_metadata(region)
if metadata['grid'] != 'predefined':
 if metadata['minlon'] < 0:metadata['minlon'] = metadata['minlon'] + 360
 if metadata['maxlon'] < 0:metadata['maxlon'] = metadata['maxlon'] + 360
#metadata['minlon'] = metadata['minlon']+360.0
#metadata['maxlon'] = metadata['maxlon']+360.0
os.system('mkdir -p %s' % metadata['dir'])
mdfile = '%s/metadata.pck' % metadata['dir']
pickle.dump(metadata,open(mdfile,'wb'))

import netCDF4 as nc
import numpy as np
import geospatialtools.gdal_tools as gdal_tools
import scipy.stats
import scipy.special
#Cl SiCl SaCl ClLo SiClLo SaClLo Lo SiLo SaLo Si LoSa Sa
#'hec'->heavy clay (1)
#'sic'->silty clay (2)
#'lic'->(light) clay (3)
#'sicl'->silty clay loam (4)
#'cl'->clay loam (5)
#'si'->silt (6)
#'sil'->silt loam (7)
#'sac'->sandy clay (8)
#'l'->loam (9)
#'sacl'->sandy clay loam (10)
#'sal'->sandy loam (11)
#'ls'->loamy sand (12)
#'s'->sand (13)
#  tile_names              =  'hec',   'sic',   'lic',   'sicl',  'cl',    'si',    'sil',   'sac',   'l',     'sacl',  'sal',  'ls',     's',     'u'
dat_w_sat = [0.468,   0.468,   0.468,   0.464,   0.465,   0.476,   0.476,   0.406,   0.439,   0.404,   0.434,   0.421,   0.339,   0.439]
dat_k_sat_ref = [0.00097, 0.0013,  0.00097, 0.002,   0.0024,  0.0028,  0.0028,  0.0072,  0.0033,  0.0045,  0.0052,  0.014,   0.047,   0.0033]
dat_psi_sat_ref = [-0.47,   -0.32,   -0.47,   -0.62,   -0.26,   -0.76,   -0.76,   -0.098,  -0.35,   -0.13,   -0.14,   -0.036   -0.069,  -0.35]
dat_chb = [12.0,    10.0,    12.0,    8.7,     8.2,     5.3,     5.3,     11.0,    5.2,     6.8,     4.7,     4.3,     2.8,     5.2]
dat_heat_capacity_dry = [1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6,   1.1e6]
dat_thermal_cond_dry  = [0.21,    0.21,    0.21,    0.21,    0.21,    0.21,    0.21,    0.21,    0.21,    0.21,    0.21,    0.21,    0.14,    0.21]
dat_thermal_cond_sat  = [1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     2.3,     1.5]
dat_thermal_cond_exp    =  [6,       6,       6,       5,       5,       5,       5,       6,       5,       5,       5,       5,       3,       5]
dat_thermal_cond_scale  =  [10,      10,      10,      0.5,     0.5,     0.5,     0.5,     10,      0.5,     0.5,     0.5,     0.5,     15,      0.5]
dat_thermal_cond_weight =  [0.7,     0.7,     0.7,     0.7,     0.7,     0.7,     0.7,     0.7,     0.7,     0.7,     0.7,     0.7,     0.2,     0.7]
dat_emis_dry            =  [1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0]
dat_emis_sat            =  [1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0]
dat_tf_depr             =  [2.0,     2.0,     2.0,     2.0,     2.0,     2.0,     2.0,     2.0,     2.0,     2.0,     2.0,     2.0,     2.0,     2.0]
dat_z0_momentum         =  [0.01,    0.01,    0.01,    0.01,    0.01,    0.01,    0.01,    0.01,    0.01,    0.01,    0.01,    0.01,    0.01,    0.01]

'''dat_w_sat=[0.380, 0.445, 0.448, 0.412, 0.414, 0.446, 0.424, 0.445, 0.445, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_awc_lm2=[0.063, 0.132, 0.109, 0.098, 0.086, 0.120, 0.101, 0.445, 0.150, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_k_sat_ref=[0.021, .0036, .0018, .0087, .0061, .0026, .0051, .0036, .0036, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_psi_sat_ref=[-.059, -0.28, -0.27, -0.13, -0.13, -0.27, -0.16, -0.28, -0.28, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_chb=[3.5,   6.4,  11.0,   4.8,   6.3,   8.4,   6.3,   6.4,   6.4, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_heat_capacity_dry=[1.2e6, 1.1e6, 1.1e6, 1.1e6, 1.1e6, 1.1e6, 1.1e6, 1.4e6,   1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_thermal_cond_ref=[1.5,0.8,1.35,  1.15, 1.475, 1.075, 1.217,  0.39, 2.e-7, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_thermal_cond_dry=[0.14,  0.21,  0.20,  .175, 0.170, 0.205, 0.183,  0.05, 2.e-7, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_thermal_cond_sat=[2.30,  1.50,  1.50,  1.90, 1.900, 1.500, 1.767,  0.50, 2.e-7, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_thermal_cond_scale=[15.0,  0.50,   10.,  2.74,  12.2,  2.24,  4.22,   1.0,   1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_thermal_cond_exp=[3.0,   5.0,   6.0,   4.0,   4.5,   5.5, 4.667,   1.0,   1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_thermal_cond_weight=[0.20,  0.70,   0.7,  0.45, 0.450, 0.700, 0.533,   1.0,   1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_emis_dry=[0.950, 0.950, 0.950, 0.950, 0.950, 0.950, 0.950, 0.950,   1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_emis_sat=[0.980, 0.975, 0.970, .9775, 0.975, .9725, 0.975, 0.975,   1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_z0_momentum=[0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01, 0.045, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_tf_depr=[0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,   0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dat_refl_dry_dir=np.array([[0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 999.0, 5*0.0],
                  [0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 999.0, 5*0.0]])
dat_refl_dry_dif=np.array([[0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 999.0, 5*0.0],
                  [0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 999.0, 5*0.0]])
dat_refl_sat_dir=np.array([[0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 999.0, 5*0.0],
                  [0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 999.0, 5*0.0]]) 
dat_refl_sat_dif=np.array([[0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 999.0, 5*0.0],
                  [0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 999.0, 5*0.0]]) '''
dat_awc_lm2 = 0.1
rsa_exp_global = 1.5
dat_refl_sat_dif = [0.333,0.333]
dat_refl_sat_dir = [0.333,0.333]
dat_refl_dry_dif = [0.333,0.333]
dat_refl_dry_dir = [0.333,0.333]

def Extract_Soil_Properties_Original(lat,lon,metadata):

 file = '/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/ground_type.nc'
 #Read in the soil type
 fp = nc.Dataset(file)
 lons = fp.variables['lon'][:]
 lats = fp.variables['lat'][:]
 zax1_10 = fp.variables['zax1_10'][:]
 frac = fp.variables['frac'][:]

 #Change to -180 to 180 if necessary
 lons[lons > 180] = lons[lons > 180] - 360

 #Find the match
 ilat = np.argmin(np.abs(lats-lat))
 ilon = np.argmin(np.abs(lons-lon))
 frac = frac[:,ilat,ilon]
 frac[8:] = 0.0
 soil_type = int(zax1_10[np.argmax(frac)])

 output = {}
 for it in xrange(metadata['fhillslope']['NN']):
  tmp = {}
  #Assign the soil information
  tmp['dat_w_sat'] = dat_w_sat[soil_type-1]
  tmp['dat_awc_lm2'] = dat_awc_lm2[soil_type-1]
  tmp['dat_k_sat_ref'] = dat_k_sat_ref[soil_type-1]
  tmp['dat_psi_sat_ref'] = dat_psi_sat_ref[soil_type-1]
  tmp['dat_chb'] = dat_chb[soil_type-1]
  tmp['dat_heat_capacity_dry'] = dat_heat_capacity_dry[soil_type-1]
  tmp['dat_thermal_cond_dry'] = dat_thermal_cond_dry[soil_type-1]
  tmp['dat_thermal_cond_sat'] = dat_thermal_cond_sat[soil_type-1] 
  tmp['dat_thermal_cond_exp'] = dat_thermal_cond_exp[soil_type-1]
  tmp['dat_thermal_cond_scale'] = dat_thermal_cond_scale[soil_type-1]
  tmp['dat_thermal_cond_weight'] = dat_thermal_cond_weight[soil_type-1]
  tmp['dat_emis_dry'] = dat_emis_dry[soil_type-1]
  tmp['dat_emis_sat'] = dat_emis_sat[soil_type-1]
  tmp['dat_z0_momentum'] = dat_z0_momentum[soil_type-1]
  tmp['dat_tf_depr'] =  dat_tf_depr[soil_type-1]
  tmp['dat_refl_dry_dir'] = dat_refl_dry_dir#dat_refl_dry_dir[:,soil_type-1]
  tmp['dat_refl_dry_dif'] = dat_refl_dry_dif#dat_refl_dry_dif[:,soil_type-1]
  tmp['dat_refl_sat_dir'] = dat_refl_sat_dir#dat_refl_sat_dir[:,soil_type-1] 
  tmp['dat_refl_sat_dif'] = dat_refl_sat_dif#dat_refl_sat_dif[:,soil_type-1] 
  tmp['rsa_exp_global'] = rsa_exp_global
  #Create if necessary the variable in the output
  if tmp.keys()[0] not in output:
   for var in tmp:
    output[var] = []
  #Add the variables
  for var in tmp:
   output[var].append(tmp[var])

 #Convert all to arrays
 for var in output:
  output[var] = np.array(output[var])
  if len(output[var].shape) == 2: output[var] = output[var].T
 
 return output


def Extract_Soil_Properties_POLARIS(lat,lon,file):

 #Open access to the file
 fp = nc.Dataset(file)

 #Read in the data
 dp = {}
 for var in ['sand_mean','silt_mean','clay_mean','smcmax_mean','smcfc_mean','smcwlt_mean','ksat_mean']:
  dp[var] = np.flipud(fp.variables[var][0,:,:])

 #Read in the hru map
 hrus = gdal_tools.read_raster('tmp/workspace/hrus.tif')

 #Assign the soil information
 uhrus = np.unique(hrus)[1::]
 output = {'dat_w_sat':[]}
 for hru in uhrus:
  m = hrus == hru
  #plt.imshow(m)
  #plt.show()
  output['dat_w_sat'].append(np.mean(dp['smcmax_mean'][m]))
 print(output['dat_w_sat'])
 exit()
 '''output['dat_awc_lm2'] = dat_awc_lm2[soil_type-1]
 output['dat_k_sat_ref'] = dat_k_sat_ref[soil_type-1]
 output['dat_psi_sat_ref'] = dat_psi_sat_ref[soil_type-1]
 output['dat_chb'] = dat_chb[soil_type-1]
 output['dat_heat_capacity_dry'] = dat_heat_capacity_dry[soil_type-1]
 output['dat_thermal_cond_dry'] = dat_thermal_cond_dry[soil_type-1]
 output['dat_thermal_cond_sat'] = dat_thermal_cond_sat[soil_type-1] 
 output['dat_thermal_cond_exp'] = dat_thermal_cond_exp[soil_type-1]
 output['dat_thermal_cond_scale'] = dat_thermal_cond_scale[soil_type-1]
 output['dat_thermal_cond_weight'] = dat_thermal_cond_weight[soil_type-1]
 output['dat_emis_dry'] = dat_emis_dry[soil_type-1]
 output['dat_emis_sat'] = dat_emis_sat[soil_type-1]
 output['dat_z0_momentum'] = dat_z0_momentum[soil_type-1]
 output['dat_tf_depr'] =  dat_tf_depr[soil_type-1]
 output['dat_refl_dry_dir'] = dat_refl_dry_dir[:,soil_type-1]
 output['dat_refl_dry_dif'] = dat_refl_dry_dif[:,soil_type-1]
 output['dat_refl_sat_dir'] = dat_refl_sat_dir[:,soil_type-1] 
 output['dat_refl_sat_dif'] = dat_refl_sat_dif[:,soil_type-1] 
 output['rsa_exp_global'] = rsa_exp_global'''
 
 return output

def compute_soil_e_depth_ksat(vars,m2):

 #top layer
 thetas = np.mean(vars['ThetaS_0cm'][m2])
 theta33 = np.mean(vars['Theta33_0cm'][m2])
 theta1500 = np.mean(vars['Theta1500_0cm'][m2])
 chb = B(theta33,theta1500)
 psisat = -10.1972*Psisat(theta33,theta1500,thetas,0)/100.0 #(Kpa->cm->meters)
 ksat_0 = Ksat(thetas,theta33,chb)/3600.0 #mm/s
 #bottom layer
 thetas = np.mean(vars['ThetaS_200cm'][m2])
 theta33 = np.mean(vars['Theta33_200cm'][m2])
 theta1500 = np.mean(vars['Theta1500_200cm'][m2])
 chb = B(theta33,theta1500)
 psisat = -10.1972*Psisat(theta33,theta1500,thetas,0)/100.0 #(Kpa->cm->meters)
 ksat_200 = Ksat(thetas,theta33,chb)/3600.0 #mm/s

 return ksat_0,ksat_200

def Extract_Soil_Properties(cdir,metadata):

 #Read in the hru map
 hrus = gdal_tools.read_raster('%s/soil_tiles_ea.tif' % cdir)

 #Read in the soil properties
 vs = ['ThetaS_0cm','soiltexture_0cm','Theta33_0cm','Theta1500_0cm',
       'ThetaS_200cm','Theta33_200cm','Theta1500_200cm']#'PsiSat','Chb']
 #HAVE TO READ IN ALL LAYERS TO COMPUTE EDECAY SOIL DEPTH
 vars = {}
 for var in vs:
  #file = '%s/%s_latlon.tif' % (cdir,var)
  file = '%s/%s_ea.tif' % (cdir,var)
  tmp = gdal_tools.read_raster(file)
  tmp = np.ma.masked_array(tmp,tmp==-9999)
  vars[var] = tmp
 #vars['ThetaS'] = vars['ThetaS_0cm']
 #vars['Theta33'] = vars['Theta33_0cm']
 #vars['Theta1500'] = vars['Theta1500_0cm']
 #vars['soiltexture'] = vars['soiltexture_0cm']
 #Eventually we should be able to include vertical soil properties
 vars['ThetaS'] = vars['ThetaS_200cm']
 vars['Theta33'] = vars['Theta33_200cm']
 vars['Theta1500'] = vars['Theta1500_200cm']
 vars['soiltexture'] = vars['soiltexture_0cm']
 #file = '%s/FAOtexture_ea.tif' % cdir
 #tclass = gdal_tools.read_raster(file)
 #tclass = np.ma.masked_array(tclass,tclass==-9999)

 #Calculate the mode of the soil types and assign it to each hru
 #Assign the properties to each hru
 uhrus = np.unique(hrus)
 uhrus = uhrus[uhrus != -9999]
 output = {}
 for hru in uhrus:
  m = hrus == hru
  #soil_type = int(scipy.stats.mode(vars['FAOtexture'][m])[0][0])
  soil_type = int(scipy.stats.mode(vars['soiltexture'][m])[0][0])
  m2 = m & (vars['ThetaS'] != -9999)
  if np.sum(m2) > 0:
   thetas = np.mean(vars['ThetaS'][m2])
   theta33 = np.mean(vars['Theta33'][m2])
   theta1500 = np.mean(vars['Theta1500'][m2])
   #ksat = np.mean(vars['Ksat'][m2])
   #Compute Brooks-Corey parameters
   chb = B(theta33,theta1500)
   psisat = -10.1972*Psisat(theta33,theta1500,thetas,0)/100.0 #(Kpa->cm->meters)
   #Compute Ksat
   ksat = Ksat(thetas,theta33,chb)/3600.0 #mm/s
   #Compute soil_e_depth
   (ksat_0,ksat_200) = compute_soil_e_depth_ksat(vars,m2)
   
   #psisat = np.mean(vars['PsiSat'][m2])
   #chb = np.mean(vars['Chb'][m2])
   #print 'thetas',thetas#np.mean(vars['ThetaS'][m2]),np.median(vars['ThetaS'][m2])
   #print 'ksat',ksat#np.mean(vars['Ksat'][m2]),np.median(vars['Ksat'][m2])
   #print 'psisat',psisat#np.mean(vars['PsiSat'][m2]),np.median(vars['PsiSat'][m2])
   #print 'chb',chb#np.mean(vars['Chb'][m2]),np.median(vars['Chb'][m2])
  else:
   thetas = 0.40
   ksat = 0.0030
   psisat = -0.0590
   chb = 3.50
   ksat_0 = 0.0030
   ksat_200 = 0.0030

  #Fallback
  if soil_type == -9999:soil_type = 1#ARBITRARY

  #Set up some QC
  if thetas < 0.2:thetas = 0.2
  if thetas > 0.7:thetas = 0.7
  if ksat < 10**-10:ksat = 10**-10
  if ksat > 1.0:ksat = 1.0
  if chb < 1:chb = 1
  if chb > 20:chb = 20
  if psisat < -10.0:psisat = -10.0
  if psisat > -10**-3:psisat = -10**-3

  #Assign the parameters
  tmp = {}
  #tmp['dat_w_sat'] = dat_w_sat[soil_type-1]
  tmp['dat_w_sat'] = thetas
  tmp['dat_awc_lm2'] = dat_awc_lm2#[soil_type-1]
  #tmp['dat_k_sat_ref'] = dat_k_sat_ref[soil_type-1]
  tmp['dat_k_sat_ref'] = ksat
  #tmp['dat_psi_sat_ref'] = dat_psi_sat_ref[soil_type-1]
  tmp['dat_psi_sat_ref'] = psisat
  #tmp['dat_chb'] = dat_chb[soil_type-1]
  tmp['dat_chb'] = chb
  tmp['dat_heat_capacity_dry'] = dat_heat_capacity_dry[soil_type-1]
  tmp['dat_thermal_cond_dry'] = dat_thermal_cond_dry[soil_type-1]
  tmp['dat_thermal_cond_sat'] = dat_thermal_cond_sat[soil_type-1] 
  tmp['dat_thermal_cond_exp'] = dat_thermal_cond_exp[soil_type-1]
  tmp['dat_thermal_cond_scale'] = dat_thermal_cond_scale[soil_type-1]
  tmp['dat_thermal_cond_weight'] = dat_thermal_cond_weight[soil_type-1]
  tmp['dat_emis_dry'] = dat_emis_dry[soil_type-1]
  tmp['dat_emis_sat'] = dat_emis_sat[soil_type-1]
  tmp['dat_z0_momentum'] = dat_z0_momentum[soil_type-1]
  tmp['dat_tf_depr'] =  dat_tf_depr[soil_type-1]
  tmp['dat_refl_dry_dir'] = dat_refl_dry_dir#dat_refl_dry_dir[:,soil_type-1]
  tmp['dat_refl_dry_dif'] = dat_refl_dry_dif#dat_refl_dry_dif[:,soil_type-1]
  tmp['dat_refl_sat_dir'] = dat_refl_sat_dir#dat_refl_sat_dir[:,soil_type-1] 
  tmp['dat_refl_sat_dif'] = dat_refl_sat_dif#dat_refl_sat_dif[:,soil_type-1] 
  tmp['rsa_exp_global'] = rsa_exp_global
  tmp['ksat_0cm'] = ksat_0
  tmp['ksat_200cm'] = ksat_200
  #Create if necessary 
  if list(tmp.keys())[0] not in output:
   for var in tmp:
    output[var] = []
  #Add the variables
  for var in tmp:
   output[var].append(tmp[var])

 #Convert all to arrays
 for var in output:
  output[var] = np.array(output[var])
  if len(output[var].shape) == 2: output[var] = output[var].T

 '''for hru in hrus: 
  m = hrus == hru
  for var in vars:
   if var not in ohrus:ohrus[var] = []
   tmp = np.mean(vars[var][m])
   ohrus[var].append(tmp)
 for var in ohrus:
  ohrus[var] = np.array(ohrus[var])'''
   
 '''#Theta33,Theta1500,PsiSat,B
 #Assign properties to land model
 uhrus = np.unique(hrus)[1::]
 output = {#'dat_w_sat':ohrus['ThetaS'],#Porosity (m3/m3)
           #'dat_awc_lm2':ohrus['Theta33'] - ohrus['Theta1500'],#Available water content (m3/m3)
           #'dat_psi_sat_ref':ohrus['PsiSat'],#
           #'dat_chb':ohrus['B'],#
	   #'dat_heat_capacity_dry':1.1e6*np.ones(nhrus),
           #'dat_thermal_cond_dry':,
           #'dat_thermal_cond_sat':,
           #'dat_thermal_cond_exp':,
           #'dat_thermal_cond_scale':,
           'dat_emis_dry':0.950*np.ones(nhrus),
           'dat_emis_sat':0.975*np.ones(nhrus),
           'dat_z0_momentum':0.01*np.ones(nhrus),
           'dat_tf_depr':0.0*np.ones(nhrus),
           'dat_refl_dry_dir':0.333*np.ones(nhrus,2),
           'dat_refl_dry_dif':0.333*np.ones(nhrus,2),
           'dat_refl_sat_dir':0.333*np.ones(nhrus,2),
           'dat_refl_sat_dif':0.333*np.ones(nhrus,2),
           #'rsa_exp_global' = rsa_exp_global*np.ones(nhrus)
           }'''
     
 '''output['dat_awc_lm2'] = dat_awc_lm2[soil_type-1]
 output['dat_k_sat_ref'] = dat_k_sat_ref[soil_type-1]
 output['dat_psi_sat_ref'] = dat_psi_sat_ref[soil_type-1]
 output['dat_chb'] = dat_chb[soil_type-1]
 output['dat_heat_capacity_dry'] = dat_heat_capacity_dry[soil_type-1]
 output['dat_thermal_cond_dry'] = dat_thermal_cond_dry[soil_type-1]
 output['dat_thermal_cond_sat'] = dat_thermal_cond_sat[soil_type-1] 
 output['dat_thermal_cond_exp'] = dat_thermal_cond_exp[soil_type-1]
 output['dat_thermal_cond_scale'] = dat_thermal_cond_scale[soil_type-1]
 output['dat_thermal_cond_weight'] = dat_thermal_cond_weight[soil_type-1]
 output['dat_emis_dry'] = dat_emis_dry[soil_type-1]
 output['dat_emis_sat'] = dat_emis_sat[soil_type-1]
 output['dat_z0_momentum'] = dat_z0_momentum[soil_type-1]
 output['dat_tf_depr'] =  dat_tf_depr[soil_type-1]
 output['dat_refl_dry_dir'] = dat_refl_dry_dir[:,soil_type-1]
 output['dat_refl_dry_dif'] = dat_refl_dry_dif[:,soil_type-1]
 output['dat_refl_sat_dir'] = dat_refl_sat_dir[:,soil_type-1] 
 output['dat_refl_sat_dif'] = dat_refl_sat_dif[:,soil_type-1] 
 output['rsa_exp_global'] = rsa_exp_global'''
 
 return output

def Ksat(thetas,theta33,chb):
  ksat = 1930.0*(thetas - theta33)**(3-1/chb)
  return ksat

def B(vwc33,vwc1500):
  b = (np.log(1500) - np.log(33))/(np.log(vwc33) - np.log(vwc1500))
  return b

def Psisat(vwc33,vwc1500,vwc0,vwcr):
  b = B(vwc33,vwc1500)
  vwc = vwc33
  psi = 33
  return psi*((vwc-vwcr)/(vwc0-vwcr))**b

#Compute column weighted median for the given variable
def compute_cwa_median(var,vdir,cdir,minlon,minlat,maxlon,maxlat,log):

 import os
 data = {}
 #t = np.array([0,5,15,30,60,100]).astype(np.float32)
 #b = np.array([5,15,30,60,100,200]).astype(np.float32)
 t = np.array([0,5]).astype(np.float32)
 b = np.array([5,15]).astype(np.float32)
 dz = b - t
 for il in xrange(t.size):
  for stat in ['min','max','var','mean']:
   if stat not in data:data[stat] = []
   #Extract the data
   file_in = '%s/%s_%s_%d_%d.vrt' % (vdir,var,stat,t[il],b[il])
   file_latlon = '%s/%s_latlon.tif' % (cdir,var)
   os.system('rm -f %s' % file_latlon)
   #file_ea = '%s/%s_ea.tif' % (cdir,var)
   #os.system('rm -f %s' % file_ea)
   #Cut out the region
   os.system('gdalwarp -ot Float32 -dstnodata -9999 -te %.16f %.16f %.16f %.16f %s %s >& %s' % (minlon,minlat,maxlon,maxlat,file_in,file_latlon,log))
   #Reproject region to equal area
   #lproj = eaproj % float((maxlon+minlon)/2)
   #os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (md['minx'],md['miny'],md['maxx'],md['maxy'],eares,eares,lproj,file_latlon,file_ea,log))
   #Read in the data
   tmp = gdal_tools.read_raster(file_latlon)
   data[stat].append(np.ma.masked_array(tmp,tmp==-9999))
 for stat in data:
  data[stat] = np.array(data[stat])
 #Create a mask
 mask = data['mean'] != -9999.0
 #Compute normalized mean
 nmean = np.copy(data['mean'])
 nmean[mask] = (data['mean'][mask] - data['min'][mask])/(data['max'][mask] - data['min'][mask])
 #Compute normalized variance
 nvar = np.copy(data['mean'])
 nvar[mask] = 1/(data['max'][mask] - data['min'][mask])**2*(data['var'][mask] + data['mean'][mask]**2 - 2*data['min'][mask]*data['mean'][mask] + data['min'][mask]**2) - nmean[mask]**2
 #Compute alpha
 alpha = np.copy(data['mean'])
 alpha[mask] = ((1-nmean[mask])/nvar[mask] - (1/nmean[mask]))*nmean[mask]**2
 #Compute beta
 beta = np.copy(data['mean'])
 beta[mask] = alpha[mask]*(1/nmean[mask] - 1)
 #QC
 epsn = 10**-4
 alpha[alpha < epsn] = epsn
 beta[beta < epsn] = epsn
 #Compute sample of all distributions
 a = alpha[:,:,:]
 b = beta[:,:,:]
 #a = a[2,100,100]
 #a = b[2,100,100]
 median = np.copy(data['mean'])
 #median[mask] = scipy.special.betaincinv(a[mask],b[mask],0.5)#*scipy.special.beta(a,b)
 #Sample from the distribution
 np.random.seed(0)
 median[mask] = np.random.beta(a[mask],b[mask])
 median[mask] = median[mask]*(data['max'][mask]-data['min'][mask]) + data['min'][mask]
 #Mask out undef
 median = np.ma.masked_array(median,median==-9999)
 #Compute the vertical average
 w = dz/np.sum(dz)
 data = np.sum(w[:,np.newaxis,np.newaxis]*median,axis=0)

 return data

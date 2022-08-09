#import netCDF4 as nc
import numpy as np
import geospatialtools.gdal_tools as gdal_tools
import scipy.stats
input_cover_types=[-1,-1,-1,-1,1,2,3,4,5,6,7,8,9]

def Extract_Vegetation_Properties_Old(lat,lon,file):

 #Read in the soil type
 fp = nc.Dataset(file)
 lons = fp.variables['lon'][:]
 lats = fp.variables['lat'][:]
 zax1_11 = fp.variables['zax1_11'][:]
 frac = fp.variables['frac'][:]

 #Change to -180 to 180 if necessary
 lons[lons > 180] = lons[lons > 180] - 360

 #Find the match
 ilat = np.argmin(np.abs(lats-lat))
 ilon = np.argmin(np.abs(lons-lon))
 frac = frac[:,ilat,ilon]
 frac[9:] = 0.0
 #veg_type = int(zax1_11[np.argmax(frac[:,ilat,ilon])])
 veg_type = int(zax1_11[np.argmax(frac)])
 #print 'veg_type',lat,lon,veg_type

 #Assign the land cover information information
 output = {}
 output['vegn'] = input_cover_types.index(veg_type)+1

 #Fallback
 if output['vegn'] == -9999:output['vegn'] = 6
 
 return output

def Extract_Vegetation_Properties(cdir,metadata):

 if metadata['landcover']['type'] == 'original':

  output = {}
  for i in  xrange(metadata['hillslope']['NN']):
   #Assign the land cover information information
   tmp = {}
   tmp['vegn'] = 1#-9999
   tmp['landuse'] = 3#-9999
   tmp['irrigation'] = 0#-9999
   tmp['tas'] = -9999
   tmp['prec'] = -9999
   #Create if necessary the variable in the output
   if list(tmp.keys())[0] not in output:
    for var in tmp:
     output[var] = []
   #Add the variables
   for var in tmp:
    output[var].append(tmp[var])

 else:

  #Read in the hru map
  #hrus = gdal_tools.read_raster('%s/soil_tiles_latlon.tif' % cdir)
  hrus = gdal_tools.read_raster('%s/soil_tiles_ea.tif' % cdir)

  #Read in the necessary data
  data = {}
  for var in ['vegetation','landuse','irrigation','pann','tann','cheight','maxlai']:
   #data[var] = gdal_tools.read_raster('%s/%s_latlon.tif' % (cdir,var))
   data[var] = gdal_tools.read_raster('%s/%s_ea.tif' % (cdir,var))
   data[var] = np.ma.masked_array(data[var],data[var]==-9999.0)

  #Extract and assign the geohydrology information
  #output = {}
  #for hru in xrange(1,int(np.max(hrus))+1):
  uhrus = np.unique(hrus)
  uhrus = uhrus[uhrus != -9999]
  output = {}
  for hru in uhrus:
   m = hrus == hru
   #Create the temporary dictionary
   #print 'vegn',int(scipy.stats.mode(data['vegetation'][m])[0][0])
   #print 'lu',int(scipy.stats.mode(data['landuse'][m])[0][0])
   tmp = {}
   tmp['vegn'] = int(scipy.stats.mode(data['vegetation'][m])[0][0])
   if tmp['vegn'] == -9999:tmp['vegn'] = 1
   tmp['landuse'] = int(scipy.stats.mode(data['landuse'][m])[0][0])
   if tmp['landuse'] == -9999:tmp['landuse'] = 3
   tmp['irrigation'] = int(scipy.stats.mode(data['irrigation'][m])[0][0])
   if tmp['irrigation'] == -9999:tmp['irrigation'] = 0
   tmp['tann'] = np.mean(data['tann'][m])
   if tmp['tann'] == -9999:tmp['tann'] = 15.0 #C
   tmp['pann'] = np.mean(data['pann'][m])
   if tmp['pann'] == -9999:tmp['pann'] = 800 #mm
   #Construct biomass
   maxlai = np.ma.getdata(np.mean(data['maxlai'][m]))
   cheight = np.ma.getdata(np.mean(data['cheight'][m]))
   species = tmp['vegn']
   if maxlai == -9999:maxlai = 0.0
   if cheight == -9999:cheight = 0.0
   bdata = calculate_biomass_components(maxlai,cheight,species)
   #print species,maxlai,cheight,bdata
   #Assign properties
   #Set c3/c4 grasses to 0...
   if ((species == 0) | (species == 1)):
    tmp['bl'] = 0.0
    tmp['br'] = 0.0
    tmp['bsw'] = 0.0
    tmp['bwood'] = 0.0
   else:
    tmp['bl'] = bdata['bl']
    tmp['br'] = bdata['br']
    tmp['bsw'] = bdata['bsw']
    tmp['bwood'] = bdata['bwood']

   #Set constraints
   init_cohort_bl = 0.05
   init_cohort_br = 0.05
   init_cohort_bsw = 0.05
   init_cohort_bwood = 0.05
   if tmp['bl'] < init_cohort_bl:tmp['bl'] = init_cohort_bl
   if tmp['br'] < init_cohort_br:tmp['br'] = init_cohort_br
   if tmp['bsw'] < init_cohort_bsw:tmp['bsw'] = init_cohort_bsw
   if tmp['bwood'] < init_cohort_bwood:tmp['bwood'] = init_cohort_bwood
  
   #Create if necessary the variable in the output
   if list(tmp.keys())[0] not in output:
    for var in tmp:
     output[var] = []
   #Add the variables
   for var in tmp:
    output[var].append(tmp[var])

 '''#Convert all to arrays
 for var in output:
  output[var] = np.array(output[var])

 if metadata['landcover']['type'] != 'original':

  #Determine if grasses and pastures are C3/C4
  #LU_PAST    = 1, & ! pasture
  #LU_CROP    = 2, & ! crops
  #LU_NTRL    = 3, & ! natural vegetation
  #LU_SCND    = 4    ! secondary vegetation
  #SP_C4GRASS   = 0, & ! c4 grass
  #SP_C3GRASS   = 1, & ! c3 grass
  #SP_TEMPDEC   = 2, & ! temperate deciduous
  #SP_TROPICAL  = 3, & ! non-grass tropical
  #SP_EVERGR    = 4    ! non-grass evergreen
  m0 = (output['landuse'] == 1) | (output['landuse'] == 3) #Natural or pasture
  m = m0 & ((output['vegn'] == 0) | (output['vegn'] ==  1))
 
  # Rule based on analysis of ED global output; equations from JPC, 2/02
  temp = output['tas'] + 273.15 #Temperature from input climate data (deg K)
  precip = output['prec'] #Precip from input climate data (mm/yr)
  pc4=np.exp(-0.0421*(273.16+25.56-temp)-(0.000048*(273.16+25.5-temp)*precip));
  pt = np.ones(temp.size) #Initialize to C3
  pt[pc4 > 0.5] = 0 #Set to C4
  output['vegn'][m] = pt[m]'''
  
 return output

def calculate_biomass_components(lai,cheight,species):

 #calculate bl
 bl = lai2bl(lai,species)
 #Compute living biomass fractions
 bliving_fractions = compute_bliving_fractions(cheight,species)
 #Compute bliving
 bliving = bl/bliving_fractions['Pl']
 #Compute other living biomass components
 br = bliving_fractions['Pr']*bliving
 bsw = bliving_fractions['Psw']*bliving
 #Compute btotal
 btotal = cheight2btotal(cheight)
 #Set some constraints
 if btotal < bliving:btotal = bliving
 #Compute bwood
 bwood = btotal - bliving
 bdata = {'bwood':abs(bwood),'bl':abs(bl),'br':abs(br),'bsw':abs(bsw),
          'btotal':abs(btotal),'bliving':abs(bliving)}

 return bdata

def lai2bl(lai,species):

 C2B = 2.0
 sp_pars = compute_species_parameters(species)
 #calculate specific leaf area
 leaf_life_span = 12.0/sp_pars['alpha']
 #calculate specific leaf area (cm2/g(biomass))
 #Global Raich et al 94 PNAS pp 13730-13734
 sla = 10.0**(2.4 - 0.46*np.log10(leaf_life_span))
 #convert to (m2/kg(carbon)
 sla = C2B*sla*1000.0/10000.0
 bl = lai/sla

 return bl

def cheight2btotal(cheight):

 if cheight > 24.19:cheight = 24.18
 return -np.log(1.0 - cheight/24.19)/0.19

def compute_bliving_fractions(cheight,species):

 #derived
 sp_pars = compute_species_parameters(species)
 #properties
 D = 1/(1 + sp_pars['c1'] + cheight*sp_pars['c2'])
 Pl = D
 Pr = sp_pars['c1']*D
 Psw = 1 - Pl - Pr
 Psw_alphasw = sp_pars['c3']*sp_pars['alpha']*D

 return {'Pl':Pl,'Pr':Pr,'Psw':Psw,'Psw_alphasw':Psw_alphasw}

def compute_species_parameters(species):

 #cs
 dc1 = np.array([1.358025,     2.222222,     0.4807692,    0.3333333,    0.1948718])
 dc2 = np.array([0.4004486,    0.4004486,    0.4004486,    0.3613833,    0.1509976])
 dc3 = np.array([0.5555555,    0.5555555,    0.4423077,    1.230769,     0.5897437])
 c1 = dc1[species]
 c2 = dc2[species]
 c3 = dc3[species]

 #alpha
 dalpha = np.array([
         [0.0,          0.0,          0.0,          0.0,          0.0],#reproduction
         [0.0,          0.0,          0.0,          0.0,          0.0],#sapwood
         [1.0,          1.0,          1.0,          0.8,         0.12],#leaf
         [0.9,         0.55,          1.0,          0.8,          0.6],#root
         [0.0,          0.0,          0.0,          0.0,          0.0],#virtual leaf
         [0.0,          0.0,        0.006,        0.012,        0.006],#structural
        ])
 CMPT_LEAF = 2
 alpha = dalpha[CMPT_LEAF,species]

 #Combine all properties
 output = {'c1':c1,'c2':c2,'c3':c3,
           'alpha':alpha}

 return output

def Compute_Properties():

 #Part 0 
 #Determine the landuse for each grid cell
 #Determine the vegetation type for each grid cell
 #Part 1
 #Need to determine if the species is:
 #A. C3 grass -> 
 #B. C4 grass
 #C. Temperate deciduous
 #D. Tropical
 #E. Evergreen
 #How? NLCD and Cropland database
 #Read in cropland database
 #Part 2
 #Determine the biomass of the species
 #Read in canopy height
 #Read in maximum LAI
 #Compute the maps of the different parameters

 return

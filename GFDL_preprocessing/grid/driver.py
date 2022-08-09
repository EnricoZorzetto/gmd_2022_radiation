import pickle
import os
import netCDF4 as nc
import numpy as np
import sys
import json
mdfile = sys.argv[1]
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))
minlat = metadata['minlat']
maxlat = metadata['maxlat']
minlon = metadata['minlon']
maxlon = metadata['maxlon']
res = metadata['res']
nlat = (maxlat - minlat)/res
nlon = (maxlon - minlon)/res
print(minlon,maxlon,minlat,maxlat,nlon,nlat)
bdir = '%s/bin' % os.getcwd()

#Change directory
os.system('rm -rf %s/tmp_gs' % metadata['dir'])
os.system('mkdir -p %s/tmp_gs' % metadata['dir'])
os.chdir('%s/tmp_gs' % metadata['dir'])

#Create the grid spec
str = '%s/make_hgrid --grid_type regular_lonlat_grid --nxbnd 2 --nybnd 2 --xbnd %f,%f --ybnd %f,%f --nlon %d --nlat %d' % (bdir,minlon,maxlon,minlat,maxlat,2*nlon,2*nlat)
os.system(str)

str = '%s/make_solo_mosaic --num_tiles 1 --dir ./' % bdir
os.system(str)

rfile = '%s/river/hydrography.tile1.nc' % metadata['dir']
str = '%s/make_quick_mosaic --input_mosaic solo_mosaic.nc --land_frac_file %s --land_frac_field land_frac' % (bdir,rfile)
os.system(str)

#Create the topographic map
file = 'topog.nc'
fp = nc.Dataset(file,'w')
fp.createDimension('ntile',1)
fp.createDimension('nx',nlon)
fp.createDimension('ny',nlat)
fpr = nc.Dataset(rfile)
land_frac = fpr.variables['land_frac'][:]
#print land_frac
fpr.close()
frac = np.copy(land_frac)
tmp = np.copy(land_frac)
#print tmp
frac[tmp > 0] = 0
frac[tmp <= 0] = 1
variable = fp.createVariable('depth','f8',('ny','nx'))
variable.standard_name = "topographic depth at T-cell centers"
variable.units = "meters"
variable[:] = frac
fp.close()

#Rename mosaic.nc to grid_spec.nc
os.system('mv mosaic.nc grid_spec.nc')

#If there are no ocean cells, then adjust the grid_spec.nc
#ncks = '/lustre/f1/unswept/Nathaniel.Chaney/miniconda2_alt/bin/ncks'
ncks = 'ncks'
if np.max(frac) == 0:os.system('%s -x -O -v lXo_file,aXo_file grid_spec.nc grid_spec.nc' % ncks)

#Move all the grid spec files to the workspace
os.system('rm -rf %s/grid_spec' % metadata['dir'])
os.system('mkdir -p %s/grid_spec' % metadata['dir'])
os.system('cp *.nc %s/grid_spec/.' % metadata['dir'])
os.chdir('%s/grid_spec' % metadata['dir'])
os.system('tar -cvf grid_spec.tar ./*.nc')
os.system('mv grid_spec.tar ..')
os.chdir('%s' % metadata['dir'])

#Rename the files
os.system('cp %s/grid_spec/horizontal_grid.nc %s/grid_spec/grid.tile1.nc' % (metadata['dir'],metadata['dir']))
os.system('cp %s/grid_spec/land_mask.nc %s/grid_spec/land_mask.tile1.nc' % (metadata['dir'],metadata['dir']))


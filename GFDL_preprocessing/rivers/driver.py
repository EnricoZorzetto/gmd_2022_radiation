import pickle
import os
import netCDF4 as nc
import sys
import json
#bdir = '/lustre/f1/unswept/Nathaniel.Chaney/projects/AMS2017/Simulations/LM-preprocessing/bin'
#bdir = '/lustre/f1/unswept/Nathaniel.Chaney/miniconda2/bin'
#bdir = '/lustre/f1/unswept/Nathaniel.Chaney/miniconda2_alt/bin'
#bdir = '/lustre/f1/unswept/Nathaniel.Chaney/miniconda_cdo/bin'
mdfile = sys.argv[1]
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))
rdir = '%s/river' % metadata['dir']
os.system('mkdir -p %s' % rdir)
file = '%s/hydrography.tile1.nc' % rdir
tmp = '%s/tmp.nc' % rdir
#file = '%s/river_data.nc' % rdir
#file_org = '/lustre/f1/unswept/Nathaniel.Chaney/projects/LM-preprocessing/datasets/river_data.nc'
#file_org = '/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170213.nc'
#ncks -C -O -x -v bname_gage,mainuse_def,igageloc,jgageloc /lustre/f1/unswept/Marjolein.VanHuijgevoort/projects/experiments/data/CONUS_0.25degree/hydrography.0.25deg_conus_20170405_resdepth.nc /lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170406.nc
#file_org = '/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170202.nc'
#file_org = '/lustre/f1/unswept/Nathaniel.Chaney/data/hydrography/hydrography.0.25deg_conus_20170406.nc'
#file_org = '/lustre/f1/unswept/Marjolein.VanHuijgevoort/projects/experiments/data/CONUS_0.25degree/hydrography.0.25deg_conus_20170405_resdepth.nc'
file_org = metadata['river_network']
minlat = metadata['minlat']
maxlat = metadata['maxlat']
minlon = metadata['minlon']
maxlon = metadata['maxlon']
if minlon < 0:minlon = minlon + 360
if maxlon < 0:maxlon = maxlon + 360
res = metadata['res']
print(minlat,maxlat,minlon,maxlon,res)
#Remove previous file
os.system('rm -f %s' % tmp)
#Regrid to the desired grid
#cdo = '/lustre/f1/unswept/Nathaniel.Chaney/miniconda/bin/cdo'
#cdo = '/lustre/f1/unswept/Nathaniel.Chaney/miniconda_cdo/bin/cdo'
cdo = 'cdo'
os.system("%s remapnn,%s/grid.cdo %s '%s'" % (cdo,metadata['dir'],file_org,tmp))
#Rename lat and lon back to grid_x,grid_y (var and dim)
os.system("ncrename -v lat,grid_y -v lon,grid_x -d lat,grid_y -d lon,grid_x '%s'" % (tmp,))
#Move
os.system('mv %s %s' % (tmp,file))
#os.system('/sw/eslogin-c3/nco/4.5.2/sles11.3_gnu5.1.0/bin/ncrename -v lat,grid_y -v lon,grid_x -d lat,grid_y -d lon,grid_x %s' % (file,))
#Extract the data
#os.system('ncea -d grid_y,%f,%f -d grid_x,%f,%f %s %s' % (
#	minlat,maxlat,minlon,maxlon,file_org,file))
#Change the lake fraction to 0
#fp = nc.Dataset(file,'a')
#fp.variables['lake_frac'][:] = 0.0
#fp.close()

import numpy as np 
import pickle
import json
import os
import sys
mdfile = sys.argv[1]
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))

#Read in the metadata
dir = metadata['dir']
size = metadata['npes']

#Create shapefile directory
sdir = '%s/shapefile' % dir
os.system('rm -rf %s' % sdir)
os.system('mkdir -p %s' % sdir)

#Send off the work to the cores
print("Processing each sub region for the shape")
if size == 1:
    os.system('python shapefile/driver_core.py %s 0 1' % mdfile)
else:
    # os.system('srun -n %d ./run_mpi "python shapefile/driver_core.py %s"' % (size, mdfile))
    #EZMPI
    os.system('mpirun -n %d ./run_mpi "python shapefile/driver_core.py %s"' % (size, mdfile))

#Merge the shapefiles
print("Merging the shapefile")
shp = '%s/tmp.shp' % sdir
for rank in range(size):
    shp_cell = '%s/rank%d/grid.shp' % (sdir,rank)
    if rank == 0:
        os.system('ogr2ogr -f "ESRI Shapefile" %s %s' % (shp,shp_cell))
    else:
        os.system('ogr2ogr -f "ESRI Shapefile" -update -append %s %s -nln tmp' % (shp,shp_cell))

#Deal with the dateline
print("Wrap around the dateline (Disabled)")
#os.system('ogr2ogr -datelineoffset -wrapdateline %s/grid.shp %s/tmp.shp' % (sdir,sdir))
os.system('ogr2ogr %s/grid.shp %s/tmp.shp' % (sdir, sdir))
os.system('rm %s/tmp.*' % sdir)

import pickle
import os
import sys
import json

def create_gridspec():
    #Create the grid file for cdo
    res = metadata['res']
    xfirst = metadata['minlon']
    #fp = open('/lustre/f2/dev/Nathaniel.Chaney/%s/%s/grid.cdo' % (dir,region),'w')
    fp = open('%s/grid.cdo' % (metadata['dir']),'w')
    fp.write('gridtype = lonlat\n')
    xsize = (metadata['maxlon']-metadata['minlon'])/metadata['res']
    fp.write('xsize    = %d\n' % xsize)
    ysize = (metadata['maxlat']-metadata['minlat'])/metadata['res']
    fp.write('ysize    = %d\n' % ysize)
    xfirst = metadata['minlon']+metadata['res']/2
    fp.write('xfirst    = %f\n' % xfirst)
    fp.write('xinc    = %f\n' % metadata['res'])
    yfirst = metadata['minlat']+metadata['res']/2
    fp.write('yfirst    = %f\n' % yfirst)
    fp.write('yinc    = %f\n' % metadata['res'])
    fp.close()
    #Run the scripts
    print("Creating river's file")
    os.system('python rivers/driver.py %s' % mdfile)# >& logs/create_river_file.txt')
    print("Creating grid")
    os.system('python grid/driver.py %s' % mdfile)# >& logs/create_grid_spec.txt')
    print("Creating shapefile")
    os.system('python shapefile/driver.py %s' % mdfile)#>& logs/create_shapefile.txt')
    print("Updating river data")
    os.system('python rivers/driver_update.py %s' % mdfile)#>& logs/update_river_data.txt')
    if metadata['land_fractions'] != 'original':
        print("Updating grid spec")
        os.system('python grid/driver.py %s' % mdfile)#>& logs/update_grid_spec.txt')
    return

def predefined_gridspec():
    print("Creating shapefile")
    os.system('python shapefile/driver.py %s' % mdfile)
    return

#Define the region
#region = sys.argv[1]
#dir = sys.argv[2]

#Read in the metadata
#mdfile = '/lustre/f2/dev/Nathaniel.Chaney/%s/%s/metadata.pck' % (dir,region)
#metadata = pickle.load(open(mdfile,'rb'))
mdfile = sys.argv[1]
metadata = json.load(open(mdfile,'r'))
os.system('mkdir -p %s' % metadata['dir'])

os.system('mkdir -p logs')

if metadata['grid'] == 'predefined':
    predefined_gridspec()
    pass
    # elif metadata['grid'] == 'skipit': # EZDEV added case
    #     pass                           # EZDEV added case
else: # case = 'new' in Nate 1x1 example
    create_gridspec() 
    pass

print("Creating database")
os.system('python land/driver.py %s' % mdfile)# >& logs/create_database.txt')

'''
if 'meteorology' in metadata:
    print("Creating meteorology")
    os.system('python meteorology/driver.py %s' % mdfile)
'''

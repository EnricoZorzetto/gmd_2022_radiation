import faulthandler
faulthandler.enable()
import numpy as np
import pickle
import land
import osgeo.ogr as ogr
import os
import sys
import psutil
import multiprocessing as mp

def memory_usage():
 process = psutil.Process(os.getpid())
 return process.memory_percent()

rank = int(sys.argv[1])
size = int(sys.argv[2])
#Read in the list of cells
md = pickle.load(open('workspace/md.pck'))
cids = md['list']
dir = md['dir']
region = md['region']
dir = '/lustre/f1/Nathaniel.Chaney/%s/%s' % (dir,region)
dbfile = '%s/workspace/cid_db_org.pck' % dir
metadata = pickle.load(open('%s/metadata.pck' % dir))
ldir = '%s/land' % metadata['dir']

#Read in the metadata
dir = metadata['dir']

#Read in the database
db = pickle.load(open(dbfile))

#Define rank's land directory
ldir = '%s/land' % dir

for cid in cids[rank::size]:
 print cid
 metadata['ldir'] = db[cid]['ldir']
 metadata['bbox'] = db[cid]['bbox']
 tile = db[cid]['tile']
 y = db[cid]['y']
 x = db[cid]['x']
 id = db[cid]['id']
 print 'Before:',cid,memory_usage() 
 #p = mp.Process(target=land.create_grid_cell_database,args=(id,metadata,tile,y,x))
 #p.start()
 #p.join()
 land.create_grid_cell_database(id,metadata,tile,y,x)
 print 'After:',cid,memory_usage() 

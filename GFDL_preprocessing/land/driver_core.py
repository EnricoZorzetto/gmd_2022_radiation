import numpy as np
import pickle
import land
import osgeo.ogr as ogr
import os
import sys
import multiprocessing as mp
import json
dbfile = sys.argv[1]
mdfile = sys.argv[2]
rank = int(sys.argv[3])
size = int(sys.argv[4])
max_tries = 1

#Recompute size 
#size = size - size % 36 #CAREFUL!

#Read in the metadata
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))
dir = metadata['dir']

#Read in the database
db = pickle.load(open(dbfile,'rb'))

#Define rank's land directory
ldir = '%s/land' % dir

cids = np.array(list(db.keys()))
count = 0
for cid in cids[rank::size]:
    print(rank,cid,count)
    count += 1
    #if count < 2:continue
    metadata['ldir'] = db[cid]['ldir']
    metadata['bbox'] = db[cid]['bbox']
    tile = db[cid]['tile']
    y = db[cid]['y']
    x = db[cid]['x']
    id = db[cid]['id']
    for i in range(max_tries):
        try:
            p = mp.Process( target=land.create_grid_cell_database, args=(id, metadata, tile, y, x) )
            p.start()
            p.join()
            #land.create_grid_cell_database(id,metadata,tile,y,x)
            break
        except: 
            cdir = '%s/tile:%d,is:%d,js:%d' % (ldir, tile, y, x)
            file = '%s/land_model_input_database.h5' % (cdir,)
            os.system('rm -f %s' % file)

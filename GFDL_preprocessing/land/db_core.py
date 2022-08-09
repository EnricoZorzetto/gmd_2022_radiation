import numpy as np
import pickle
import os
import sys
import h5py
import json
mdfile = sys.argv[1]
rank = int(sys.argv[2])
size = int(sys.argv[3])

#Read in the metadata
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))
dir = metadata['dir']

#Read in the database
dbfile = '%s/workspace/cid_db_org.pck' % dir
db = pickle.load(open(dbfile,'rb'))

#Define rank's land directory
ldir = '%s/land' % dir

tid = rank + 1
file = '%s/ptiles.%s.tile%d.h5' % (dir,metadata['name'],tid)
os.system('rm -f %s' % file)
fp = h5py.File(file,'w')
gp = fp.create_group('grid_data')
#Construct database
db = pickle.load(open(dbfile,'rb'))
mdb = {}
for cid in db:
    if 'tile:%d' % tid not in cid:continue
    print(cid)
    try:
        print('%s/%s/land_model_input_database.h5' % (ldir,cid))
        fpr = h5py.File('%s/%s/land_model_input_database.h5' % (ldir,cid),'r')
        h5py.h5o.copy(fpr.id,bytes('/',encoding="utf-8"),gp.id,bytes(cid,encoding="utf-8"))
        #h5py.h5o.copy(fpr.id,'/',gp.id,cid)
        fpr.close()
    except:
        print("missing: %s" % cid)
        mdb[cid] = db[cid]

#Save the missing database
pickle.dump(mdb,open('%s/workspace/ptiles_missing_tile%d.pck' % (dir,tid),'wb'),pickle.HIGHEST_PROTOCOL)

#Close the file
fp.close()

import pickle
import os
import h5py

#Read in the metadata
metadata = pickle.load(open('metadata.pck'))
dir = metadata['dir']
size = metadata['npes']

#Create land directory
ldir = '%s/land' % dir
file = '%s/land_model_input_database.h5' % (ldir,)
os.system('rm -f %s' % file)
fp = h5py.File(file,'w')
gp = fp.create_group('grid_data')
db = pickle.load(open('workspace/cid_db_org.pck'))
for cid in db:
 print cid
 fpr = h5py.File('%s/%s/land_model_input_database.h5' % (ldir,cid),'r')
 h5py.h5o.copy(fpr.id,'/',gp.id,cid)
 fpr.close()

#Add the max number of tiles to the metadata
max_npt = 0
for cell in fp['grid_data']:
 gp = fp['grid_data'][cell]
 if len(gp['metadata']['tid'][:]) > max_npt:max_npt = len(gp['metadata']['tid'][:])
for cell in fp['grid_data']:
 fp['grid_data'][cell]['metadata']['max_npt'] = [max_npt,]

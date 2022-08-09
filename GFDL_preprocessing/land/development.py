import faulthandler
faulthandler.enable()
import pickle
from hillslope import hillslope_properties
import os
import sys
import psutil

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
log = 'workspace/log.txt'

#Read in the metadata
dir = metadata['dir']

#Read in the database
db = pickle.load(open(dbfile))

#Define rank's land directory
ldir = '%s/land' % dir

for cid in cids[rank::size]:
 print cid
 cdir = '%s/%s' % (ldir,cid)
 soil_frac = 1.0
 buffer = 100
 print 'Before:',cid,memory_usage()
 hillslope_information = hillslope_properties.Extract_Hillslope_Properties_Updated(cdir,metadata,soil_frac,buffer,log,cid)
 print 'After:',cid,memory_usage()

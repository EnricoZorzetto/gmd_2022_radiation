import os
import pickle
import osgeo.ogr as ogr
import numpy as np
import sys
import glob
import json
mdfile = sys.argv[1]
rank = int(sys.argv[2])
size = int(sys.argv[3])

#Read in the metadata
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))

#Create the directory in unswept
dir = '/lustre/f2/dev/Nathaniel.Chaney/%s/%s' % (metadata['dir'].split('/')[-2],metadata['dir'].split('/')[-1])
os.system('mkdir -p %s' % dir)

#Copy the model databases
files = glob.glob('%s/ptiles*' % metadata['dir'])
for ifile in files[rank::size]:
 ofile = '%s/%s' % (dir,ifile.split('/')[-1])
 os.system('cp %s %s' % (ifile,ofile))

#Copy the summary databases
files = glob.glob('%s/summary*' % metadata['dir'])
for ifile in files[rank::size]:
 ofile = '%s/%s' % (dir,ifile.split('/')[-1])
 os.system('cp %s %s' % (ifile,ofile))

#Copy the tile maps
tdirs = glob.glob('%s/land/*' % metadata['dir'])
odir = '%s/tmaps' % dir
os.system('mkdir -p %s' % odir)
for tdir in tdirs[rank::size]:
 ifile = '%s/tiles.tif' % (tdir,)
 ofile = '%s/%s.tif' % (odir,tdir.split('/')[-1])
 os.system('cp %s %s' % (ifile,ofile))
 print(rank,tdirs.index(tdir),len(tdirs))

import os
import pickle
import osgeo.ogr as ogr
import numpy as np
import sys
import json
mdfile = sys.argv[1]
rank = int(sys.argv[2])
size = int(sys.argv[3])
#metadata = pickle.load(open(mdfile,'rb'))
metadata = json.load(open(mdfile,'r'))
dir = metadata['dir']
ldir = '%s/land' % dir
 
#Open access to the shapefile
file_shp = '%s/shapefile/grid.shp' % dir
driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open(file_shp, 0)

layer = ds.GetLayer()
#Get all the ids
ils = []
for feature in layer:
 ils.append(feature.GetFID())

#Assemble the database
db = {}
ils = np.array(ils)
for il in ils[rank::size]:
 feature = layer[int(il)]
 id = feature.GetField("ID")
 y = feature.GetField("Y")
 x = feature.GetField("X")
 tile = feature.GetField("TILE")
 #lfrac = feature.GetField("LAND_FRAC")
 lfrac = feature.GetField("LFN")
 if lfrac == 0.0:continue
 print('id:%d,tile:%d,x:%d,y:%d' % (id,tile,x,y))
 #Retrieve coordinates of envelope
 bbox = feature.GetGeometryRef().GetEnvelope()
 #Adding info to output database
 cid = 'tile:%d,is:%d,js:%d' % (tile,y,x)
 db[cid] = {'ldir':ldir,'bbox':bbox,'x':x,'y':y,'id':id,'tile':tile,'lfrac':lfrac}

#Save the database
os.system('mkdir -p %s/workspace/org' % dir)
pickle.dump(db,open('%s/workspace/org/%d.pck' % (dir,rank),'wb'),pickle.HIGHEST_PROTOCOL)

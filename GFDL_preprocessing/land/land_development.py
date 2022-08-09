import os
import sys
import numpy as np
import pickle
import subprocess
import shlex
import psutil
import os
import time
import glob

def memory_usage(pid=os.getpid()):
 process = psutil.Process(pid)
 return process.memory_percent()

def missing(region,dir):
 output = []
 #Read the log
 fp = open('log.txt')
 for line in fp:
  if 'missing' not in line:continue
  c = line.split('\n')[0][9:]
  output.append(c)
 output = np.array(output)
 md = {}
 md['list'] = output
 md['dir'] = dir
 md['region'] = region
 pickle.dump(md,open('workspace/md.pck','w'))
 return

def misc2(region,dir):
 output = []
 #Read the log
 fp = open('/lustre/f1/Nathaniel.Chaney/predefined_input/c96_OM4_025_grid_No_mg_drag_v20160808/workspace/nid01573.txt')
 tmp = []
 for line in fp:
  if 'begin' in line:tmp.append(line[6:-1])
  if 'end' in line:
   tmp.pop(line[4:-1])
 output = []
 #Compute the info
 for cid in tmp:
  if len(glob.glob('/lustre/f1/Nathaniel.Chaney/predefined_input/c96_OM4_025_grid_No_mg_drag_v20160808/land/%s/*' % cid)) < 10:output.append(cid)
 output = np.array(output)
 print output
 md = {}
 md['list'] = output
 md['dir'] = dir
 md['region'] = region
 pickle.dump(md,open('workspace/md.pck','w'))
 return

def misc(region,dir):
 site = 'tile:1,is:1,js:1'
 #site = 'tile:1,is:172,js:340'
 #site = 'tile:1,is:172,js:346'
 #site = 'tile:1,is:174,js:286'
 #site = 'tile:1,is:5,js:7'
 #site = 'tile:1,is:8,js:11'
 #site = 'tile:1,is:23,js:39'
 #site = 'tile:1,is:1,js:20'
 #site = 'tile:1,is:53,js:104'
 #site = 'tile:1,is:48,js:81'
 #site = 'tile:1,is:48,js:75'
 #site= 'tile:1,is:58,js:69'
 #site = 'tile:1,is:40,js:74'
 #site = 'tile:1,is:6,js:77'
 #site = 'tile:2,is:192,js:43'
 #site = 'tile:2,is:185,js:91'
 #site = 'tile:5,is:55,js:5'
 #site = 'tile:1,is:50,js:72'
 #site = 'tile:3,is:50,js:37'
 #site = 'tile:5,is:36,js:10'
 #site = 'tile:3,is:31,js:59'
 #site = 'tile:3,is:10,js:308'
 #site = 'tile:2,is:382,js:84'
 #site = 'tile:3,is:13,js:18'
 #site = 'tile:3,is:66,js:42'
 #site = 'tile:3,is:51,js:72'
 #site = 'tile:1,is:36,js:92'
 #site = 'tile:3,is:95,js:27'
 #site = 'tile:3,is:95,js:21'
 #site = 'tile:3,is:51,js:75'
 #site = 'tile:3,is:27,js:53'
 #site = 'tile:3,is:27,js:57'
 #site = 'tile:1,is:12,js:5'
 #site = 'tile:1,is:12,js:91'
 #site = 'tile:1,is:13,js:2'
 #site = 'tile:1,is:29,js:92'
 #site = 'tile:1,is:37,js:75'
 #site = 'tile:1,is:13,js:9'
 #site = 'tile:1,is:50,js:67'
 #site = 'tile:1,is:87,js:57'
 #site = 'tile:2,is:37,js:87'
 #site = 'tile:1,is:69,js:52'
 #site = 'tile:1,is:60,js:82'
 #site = 'tile:5,is:60,js:12'
 #site = 'tile:6,is:64,js:44'
 md = {}
 md['list'] = np.array([site,])
 md['dir'] = dir
 md['region'] = region
 pickle.dump(md,open('workspace/md.pck','w'))
 return

#region = 'GLOBE_1DEG'
#dir = 'AMS2017'
#region = 'c96_OM4_025_uni_grid.v20150522'
#region = 'chaney2017_global'
#region = 'c192_OM4_025_grid_No_mg_drag_v20160808'
#region = 'c96_OM4_025_grid_No_mg_drag_v20160808'
region = 'chaney2017_global_dev'
#region = 'C384.v20150402_k2dh25p3'
#region = 'dev_chaney2017'
dir = 'predefined_input'
#Create the metadata
misc(region,dir)
#misc2(region,dir)

#Run the list
#os.system('aprun -n 32 run_mpi "python land/development2.py"')
#os.system('aprun -n 1 run_mpi "python land/development.py"')
os.system('python -W error land/development2.py 0 1')
#os.system('python -W error land/development.py 0 1')
exit()
#str = shlex.split('python -W error land/development2.py 0 1')
str = shlex.split('python -W error land/development.py 0 1')
proc = subprocess.Popen(str)
t = 1
while t == 1:
 print memory_usage(proc.pid)
 time.sleep(1)
proc.kill()
exit()
pid = proc.pid
print pid

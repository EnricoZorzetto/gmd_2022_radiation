import h5py
import os
import pickle
import json
import osgeo.ogr as ogr
import numpy as np
import sys
import netCDF4 as nc
mdfile = sys.argv[1]

def Add_Missing_Cells():
   
    for tid in range(1,metadata['ntiles']+1):
        print("Filling in missing cells in face %d" % tid)
        #Open access to the database
        file = '%s/ptiles.%s.tile%d.h5' % (dir, metadata['name'], tid)
        fp = h5py.File(file,'a')
      
        #Open acess to the land mask
        fplm = nc.Dataset(metadata['lm_template'].replace('$tid',str(tid)))
        lm = fplm.variables['mask'][:]
        fplm.close()
      
        #Iterate through all the cells with land fraction higher than 0
        idx = np.where(lm > 0)
        ncells = len(idx[0])
        gp = fp['grid_data']
        cells = gp.keys()
        xs = []
        ys = []
        for cell in cells:
            xs.append(int(cell.split(',')[2].split(':')[1])-1)
            ys.append(int(cell.split(',')[1].split(':')[1])-1)
        xs = np.array(xs)
        ys = np.array(ys)
        for i in range(ncells):
            x = idx[1][i]
            y = idx[0][i]
            cid = 'tile:%d,is:%d,js:%d' % (tid, y+1, x+1)
            if cid in cells:continue
            #Look fot he closest grid cell
            dist0 = ((xs - x)**2 + (ys - y)**2)
            dist = dist0[dist0 != 0]
            imin = np.where(dist0 == np.min(dist))[0][0]
            xc = xs[imin]
            yc = ys[imin]
            strc = 'tile:%d,is:%d,js:%d' % (tid,yc+1,xc+1)
         
            #Copy its database
            print('missing:%s,fill:%s' % (cid,strc))
            gp.copy(strc,gp,name=cid)
      
        #Close the file
        fp.close()
   
    return

def Create_Summary_Database(dir):
   
    #parameters
    species = {0:'C4GRASS',1:'C3GRASS',2:'TEMPDEC',3:'TROPICAL',4:'EVERGR'}
    landuse = {1:'PAST',2:'CROP',3:'NTRL',4:'SCND',5:'URBN'}
   
    for tid in range(1,metadata['ntiles']+1):
   
        print("Creating summary database for tile %s" % tid)
        hfile = metadata['rn_template'].replace('$tid',str(tid))#'%s/river/river_data.nc' % dir
        lfile = '%s/ptiles.%s.tile%d.h5' % (dir,metadata['name'], tid)
        #lfile = '%s/land/land_model_input_database.h5' % dir
        #file = '%s/land/summary.nc' % dir
        file = '%s/summary.%s.tile%d.h5' % (dir,metadata['name'], tid)
        os.system('rm -f %s' % file) # EZDEV added -f
       
        #Read in the metadata information from the hydrography file
        fp = nc.Dataset(hfile)
        lats = fp['grid_y'][:]
        lons = fp['grid_x'][:]
        nlon = lons.size
        nlat = lats.size
        data = {}
       
        #Iterate through the cells
        fp = h5py.File(lfile)
        count = 0
        for cell in fp['grid_data']:
            count += 1
            print(count,cell)
            #if (tid != 1) & (count <= 1066):continue
            #if cell == 'tile:3,is:5,js:20':continue
            #if cell != 'tile:2,is:91,js:20':continue
            #if cell == 'tile:1,is:12,js:5':continue
            #if cell == 'tile:1,is:12,js:91':continue
            #if cell == 'tile:1,is:13,js:2':continue
            #if cell != 'tile:1,is:29,js:92':continue
            #if cell == 'tile:6,is:38,js:38':continue
            #if cell == 'tile:6,is:63,js:44':continue
            #if cell == 'tile:6,is:64,js:44':continue
            #if cell != 'tile:6,is:38,js:38':continue
            gp = fp['grid_data'][cell]
            ilat = int(cell.split(',')[1].split(':')[1])-1
            ilon = int(cell.split(',')[2].split(':')[1])-1
            #Extract the desired information
            lat = gp['metadata']['latitude'][:][0]
            lon = gp['metadata']['longitude'][:][0]
            frac = np.array(gp['metadata']['frac'],ndmin=1)
            #print cell,lat,lon,ilat,ilon
            #Compute the standard deviation of the meteorological weights
            #vars = ['prec','srad','tavg','vapr','wind']
            vars = ['prec','srad','tavg','vapr','wind']
            for var in vars:
                mtmp = np.mean(np.array(gp['metadata'][var][:],ndmin=2),axis=0)
                '''if mtmp.size != frac.size:
                 tmp0 = np.ones(frac.size)
                 if mtmp.size > 0:
                  tmp0[-mtmp.size:] = mtmp
                 mtmp = tmp0'''
                if var not in data:
                    data[var] = np.zeros((nlat, nlon))
                    data[var][:] = -9999.0
                tmp0 = np.sum(frac*mtmp)
                tmp1 = np.sum(frac*(mtmp-tmp0)**2)
                data[var][ilat,ilon] = tmp1
            #Define the number of tiles
            ntiles = frac.size
            if 'ntiles' not in data:
                data['ntiles'] = np.zeros((nlat, nlon))
                data['ntiles'][:] = -9999.0
            data['ntiles'][ilat,ilon] = ntiles
            #Lake info
            if 'lake' in gp:
                lfrac = np.array(gp['lake']['frac'],ndmin=1)
                #Define the fraction of land that is lake
                if 'lfrac' not in data:
                    data['lfrac'] = np.zeros((nlat,nlon))
                    data['lfrac'][:] = -9999.0
                data['lfrac'][ilat,ilon] = np.sum(lfrac)
                print('EZDEV: lake_frac lfrac = {}, sum = {}'.format(lfrac, np.sum(lfrac)))
            #Glacier info
            if 'glacier' in gp:
                gfrac = np.array(gp['glacier']['frac'],ndmin=1)
                #Define the fraction of land that is lake
                if 'gfrac' not in data:
                    data['gfrac'] = np.zeros((nlat, nlon))
                    data['gfrac'][:] = -9999.0
                data['gfrac'][ilat,ilon] = np.sum(gfrac)
                print('EZDEV: glac_frac gfrac = {}, sum = {}'.format(gfrac, np.sum(gfrac)))
            #Soil info
            if 'soil' in gp:
                sfrac = np.array(gp['soil']['frac'],ndmin=1)
                #Define the fraction of land that is soil
                if 'sfrac' not in data:
                    data['sfrac'] = np.zeros((nlat,nlon))
                    data['sfrac'][:] = -9999.0
                data['sfrac'][ilat,ilon] = np.sum(sfrac)
                print('EZDEV: soil_frac sfrac = {}, sum = {}'.format(sfrac, np.sum(sfrac)))
                #Normalize the soil fraction
                sfrac = sfrac/np.sum(sfrac)
                #Find means
                #for var in ['dat_chb','dat_k_sat_ref','dat_w_sat','tile_hlsp_elev','tile_hlsp_length','tile_hlsp_slope','tile_hlsp_width','irrigation','dat_psi_sat_ref','bl','bsw','bwood','br']:
                for var in ['dat_chb','dat_k_sat_ref','dat_w_sat','irrigation','dat_psi_sat_ref','bl','bsw',
                               'bwood','br','tile_hlsp_slope','tile_hlsp_width','wtd',
                               'soil_depth','depth_to_bedrock','gw_perm','ksat_0cm','ksat_200cm']:
                    tmp = np.sum(sfrac*np.array(gp['soil'][var],ndmin=1))
                    if var not in data:
                        data[var] = np.zeros((nlat, nlon))
                        data[var][:] = -9999.0
                    data[var][ilat,ilon] = tmp
                #Compute hillslope properties
                vars = ['tile_hlsp_length','tile_hlsp_elev']
                for var in vars:
                    if var not in data:
                        data[var] = np.zeros((nlat, nlon))
                        data[var][:] = -9999.0
                hidx_k = gp['soil']['hidx_k'][:]
                hidx_j = gp['soil']['hidx_j'][:]
                ls = []
                es = []
                fs = []
                for hid in np.unique(hidx_k):
                    mk = hidx_k == hid
                    l = 0
                    e = 0
                    f = 0
                    for tid in np.unique(hidx_j[mk]):
                        m = hidx_j[mk] == tid
                        #print np.sum(m),np.sum(mk)
                        l = l + np.mean(np.array(gp['soil']['tile_hlsp_length'],ndmin=1)[mk][m])
                        f = f + np.sum(np.array(gp['soil']['frac'],ndmin=1)[mk][m])
                        if tid == np.max(hidx_j[mk]):e = np.mean(np.array(gp['soil']['tile_hlsp_elev'],ndmin=1)[mk][m])
                    ls.append(l)
                    es.append(e)
                    fs.append(f)
                fs = np.array(fs)
                fs = fs/np.sum(fs)
                es = np.sum(fs*es)
                ls = np.sum(fs*ls)
                data['tile_hlsp_length'][ilat, ilon] = ls
                data['tile_hlsp_elev'][ilat, ilon] = es
                #Create species fractions
                tmp = np.array(gp['soil']['vegn'],ndmin=1)
                for isp in species:
                    m = tmp == isp
                    f = np.sum(sfrac[m])
                    if species[isp] not in data:
                        data[species[isp]] = np.zeros((nlat, nlon))
                        data[species[isp]][:] = -9999.0
                    data[species[isp]][ilat, ilon] = f
                #Create landuse fractions 
                tmp = np.array(gp['soil']['landuse'],ndmin=1)
                for ilu in landuse:
                    m = tmp == ilu
                    f = np.sum(sfrac[m])
                    if landuse[ilu] not in data:
                        data[landuse[ilu]] = np.zeros((nlat, nlon))
                        data[landuse[ilu]][:] = -9999.0
                    data[landuse[ilu]][ilat, ilon] = f
        fp.close()
       
        #Output database
        fpo = nc.Dataset(file,'w')
        #Create the dimensions
        fpo.createDimension('latitude',nlat)
        fpo.createDimension('longitude',nlon)
        #latitude
        fpo.createVariable('latitude','f8',('latitude',))
        fpo['latitude'].standard_name = "latitude"
        fpo['latitude'].long_name = "latitude"
        fpo['latitude'].units = "degrees_north"
        fpo['latitude'].axis = "Y"
        fpo['latitude'][:] = lats
        #longitude
        fpo.createVariable('longitude','f8',('longitude',))
        fpo['longitude'].standard_name = "longitude"
        fpo['longitude'].long_name = "longitude"
        fpo['longitude'].units = "degrees_east"
        fpo['longitude'].axis = "X"
        fpo['longitude'][:] = lons
       
        for var in data:
            fpo.createVariable(var,'f4',('latitude','longitude'))
            fpo[var].long_name = var
            fpo[var].missing_value = -9999.0
            fpo[var].standard_name = "N/A"
            fpo[var].units = "N/A"
            fpo[var][:]  = data[var]
       
        fpo.close()
      
    return
   
def Assemble_Cell_List_Original(dir):

    #nsize = 100
    nsize = size
    #Read in the metadata
    #metadata = pickle.load(open(mdfile,'rb'))
    metadata = json.load(open(mdfile,'r'))
    dir = metadata['dir']
    ldir = '%s/land' % dir
    os.system('rm -f %s/workspace/org/*.pck' % dir)
   
    #Split up the processing
    if size == 1:
        os.system('python land/process_shapefile_core.py %s 0 1' % (mdfile,))
    else:
        # os.system('srun -n %d run_mpi "python land/process_shapefile_core.py %s"' % (nsize,mdfile,))
        #EZMPI
        os.system('mpirun -n %d run_mpi "python land/process_shapefile_core.py %s"' % (nsize,mdfile,))
      
    #Join all the pickled files to create the final dictionary
    db = {}
    for rank in range(nsize):
        #Open access to the database
        tmp = pickle.load(open('%s/workspace/org/%d.pck' % (dir,rank),'rb'))
        for cid in tmp:
            db[cid] = tmp[cid]
      
    #Save the database
    pickle.dump(db,open('%s/workspace/cid_db_org.pck' % dir,'wb'),pickle.HIGHEST_PROTOCOL)
   
    return db
   
def Construct_Database_Old():
    
    file = '%s/land_model_input_database.h5' % (ldir,)
    #fp = h5py.File(file,'a')
    #gp = fp['grid_data']
    os.system('rm -f %s' % file)
    fp = h5py.File(file,'w')
    gp = fp.create_group('grid_data')
    for i in range(1):
        if i == 0:
            print("Assembling and reading the original cell list")
            Assemble_Cell_List_Original(dir)
            dbfile = '%s/workspace/cid_db_org.pck' % dir
            #exit()#HACK
        else:
            print("Assembling and reading the updated cell list")
            dbfile = '%s/workspace/cid_db_upd.pck' % dir
        #Run the processor
        if size == 1:
            os.system('python land/driver_core.py %s %s 0 1' % (dbfile,mdfile))
        else: 
            # os.system('srun -n %d run_mpi "python land/driver_core.py %s %s"' % (size,dbfile,mdfile))
            #EZMPI
            os.system('mpirun -n %d run_mpi "python land/driver_core.py %s %s"' % (size,dbfile,mdfile))
        #Add the cell databases to the output database
        db = pickle.load(open(dbfile))
        mdb = {}
        for cid in db:
            print(cid)
            try:
                fpr = h5py.File('%s/%s/land_model_input_database.h5' % (ldir,cid),'r')
                h5py.h5o.copy(fpr.id,'/',gp.id,cid)
                fpr.close()
            except:
                print("missing: %s" % cid)
                mdb[cid] = db[cid]
        #Save the missing database
        pickle.dump(mdb,open('%s/workspace/cid_db_upd.pck' % dir,'w'),pickle.HIGHEST_PROTOCOL)
        #If the number of missing cells is 0 then exit
        if len(mdb) == 0:break
   
    #Add the max number of tiles to the metadata
    max_npt = 0
    for cell in fp['grid_data']:
        gp = fp['grid_data'][cell]
        if len(gp['metadata']['tid'][:]) > max_npt:max_npt = len(gp['metadata']['tid'][:])
    for cell in fp['grid_data']:
        fp['grid_data'][cell]['metadata']['max_npt'] = [max_npt,]
   
    #Close the file
    fp.close()
      
    return

def Process_Grid_Cells():
 
    print("Assembling and reading the original cell list")
    Assemble_Cell_List_Original(dir)
    dbfile = '%s/workspace/cid_db_org.pck' % dir
    #Empty nid workspace
    os.system('rm -rf %s/workspace/nid' % dir)
    os.system('mkdir -p %s/workspace/nid' % dir)
    #Run the processor
    if size == 1:
        os.system('python land/driver_core.py %s %s 0 1' % (dbfile, mdfile))
    else: 
        # os.system('srun -n %d run_mpi "python land/driver_core.py %s %s"' % (size, dbfile, mdfile))
        #EZMPI
        os.system('mpirun -n %d run_mpi "python land/driver_core.py %s %s"' % (size, dbfile, mdfile))
      
    return

def Construct_Database():
 
    size = metadata['ntiles']
    if size == 1:
        os.system('python land/db_core.py %s 0 1' % mdfile)
    else:
        # os.system('srun -n %d run_mpi "python land/db_core.py %s"' % (size, mdfile))
        #EZMPI
        os.system('mpirun -n %d run_mpi "python land/db_core.py %s"' % (size, mdfile))
    return

def Clean_Up():

    if size == 1:
        os.system('python land/cleanup.py %s 0 1' % mdfile)
    else:
        # os.system('srun -n %d run_mpi "python land/cleanup.py %s"' % (size, mdfile))
        #EZMPI
        os.system('mpirun -n %d run_mpi "python land/cleanup.py %s"' % (size, mdfile))
    return

#Read in the metadata
#metadata = pickle.load(open(mdfile,'rb'))
#metadata = pickle.load(open('metadata.pck'))
metadata = json.load(open(mdfile,'r'))
dir = metadata['dir']
size = metadata['npes']

#Create land directory
ldir = '%s/land' % dir
#os.system('rm -rf %s' % ldir)
os.system('mkdir -p %s' % ldir)

#Get the work done...
Process_Grid_Cells() 

#Construct the database
Construct_Database()

#Add the missing cells
Add_Missing_Cells()

#Create the summary database
Create_Summary_Database(dir)

#Clean up
#Clean_Up()

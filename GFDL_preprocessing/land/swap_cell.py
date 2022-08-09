import h5py
#Swap out a group for a new one
def swap_cell(site):
 #Open access to the database with the cell
 fin = '%s/land/%s/land_model_input_database.h5' % (dir,site)
 #First remove the cell
 #print fp['grid_data'][site]
 remove_cell(site)
 #Add the correct one
 fpr = h5py.File(fin,'r')
 h5py.h5o.copy(fpr.id,'/',fp['grid_data'].id,site)
 fpr.close()
 print fp['grid_data'][site]['metadata']['frac'][:]
 return

#Remove the grid cell
def remove_cell(site):
 del fp['grid_data'][site]
 return

dir = '/lustre/f1/Nathaniel.Chaney/predefined_input/c192_OM4_025_grid_No_mg_drag_v20160808'
file = '%s/ptiles.c192_OM4_025_grid_No_mg_drag_v20160808.tile2.h5' % dir
site = 'tile:2,is:185,js:91'

fp = h5py.File(file,'a')

#swap cell
swap_cell(site)

#remove cell
#remove_cell(site)

fp.close()

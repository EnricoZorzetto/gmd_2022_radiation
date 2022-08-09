import os
import json

output_parent_folder = os.path.join('/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/')
outfolder = os.path.join(output_parent_folder, 'gmd_2021_grids')
exp_folder = os.path.join(outfolder, 'exp_files')

# cleanup and create output folders
if os.path.exists(outfolder):
    print('cleaning up the old output directory...')
    os.system("rm -r {}".format(outfolder ))
if os.path.exists(outfolder):
    print('cleaning up the old experiment files...')
    os.system("rm -r {}".format(exp_folder ))
os.system("sleep 3")
os.makedirs(outfolder)
os.makedirs(exp_folder)
# if not os.path.exists(outfolder):
    # os.makedirs(outfolder)
# if not os.path.exists(exp_folder):
    # os.makedirs(exp_folder)

print('creating the new json files...')

# TILES CONFIGURATIONS TO EXPLORE:
# obtained varying one of k (# ch. hillslopes), n (max # height bands),
# and p (number sub-band clusters)
# use notation k4n1pV for example to have k=4 fixed, n_max = 1 fixed, and variable p
# use notation kVn1p2 for example to have k variable, n_max = 1 fixed, and p = 2 fixed
# with the range of p or k given in the range below
# do not use adaptive p - do that later only for global or large simulations
# note, when n (i.e., n_max) > 1 the number of tiles can depend on topography, not fixed
configs    = [ 'k5n1pV', 'k2n1pV', 'kVn1p2', 'kVn1p5', 'k2nVp2']
var_values = [1, 2, 5, 10, 20, 50, 100, 200]

# create one output folder for each of these simulation types
for ci in configs: 
    os.makedirs(os.path.join(outfolder,  ci)) # create output folders
    os.makedirs(os.path.join(exp_folder, ci)) # create output folders

# SIMUALTIONS TO PERFORM
domains = ['EastAlps',   'Peru',   'Nepal']
minlon  = [    12.0,      -73.0,     81.0]
maxlon  = [    13.0,      -72.0,     82.0]
minlat  = [    46.0,      -14.0,     29.0]
maxlat  = [    47.0,      -13.0,     30.0]

# COMBINATIONS OF k - p - n to explore
def get_tile_numbers(tag, var):
    k = tag[1]
    n = tag[3]
    p = tag[5]
    if k == 'V': 
        k=var
    if n == 'V': 
        n=var
    if p == 'V': 
        p=var
    return int(k), int(n), int(p)

# test
# get_tile_numbers( 'k5n1pV', 7)
# get_tile_numbers( 'k2nVp2', 12)


# load the experiment file (for example, the one for use in Gaea)
with open("experiments/experiment_gmd_2021_gaea.json", "r") as jsonFile:
    data0 = json.load(jsonFile)

# nconfigs = len(configs)
# nvarvalues = len(var_values)
# ndomains = len(domains)

list_oj_jobfiles = []

for ic, con in enumerate(configs):   
    res_folder_ic = os.path.join(outfolder, configs[ic])
    exp_folder_ic = os.path.join(exp_folder, configs[ic])
    for iv, var in enumerate(var_values):
        for idm, dom in enumerate(domains):
            mydata = data0.copy()       
            myk, myn, myp = get_tile_numbers(con, var) # based on job tag
            mydata["name"] = "res_{}_k_{}_n_{}_p_{}".format(dom, myk, myn, myp)
            mydata["dir"] = os.path.join(res_folder_ic, mydata["name"])
            mydata["minlon"] = minlon[idm]
            mydata["maxlon"] = maxlon[idm]
            mydata["minlat"] = minlat[idm]
            mydata["maxlat"] = maxlat[idm]
            mydata["hillslope"]["k"] = myk
            mydata["hillslope"]["max_nbands"] = myn
            mydata["hillslope"]["p"] = myp
            # other relevant parameters that I do not wish to tune for now
            mydata["hillslope"]["channel_threshold"] = 100000
            mydata["hillslope"]["min_nbands"] = 1 # minimum # of height bands
            mydata["hillslope"]["dh"] = 100 # m elevation difference to define height bands
            mydata["hillslope"]["use_adaptive_p"] = 0 # only for global simul (0=False, 1 = True)
            mydata["gs_template"] = os.path.join(res_folder_ic, mydata["name"], "grid_spec/grid.tile$tid.nc")
            mydata["lm_template"] = os.path.join(res_folder_ic, mydata["name"], "grid_spec/land_mask.tile$tid.nc")
            mydata["rn_template"] = os.path.join(res_folder_ic, mydata["name"], "river/hydrography.tile$tid.nc")

            with open( os.path.join(exp_folder_ic, "{}.json".format(mydata["name"])), "w") as jsonFile:
                json.dump(mydata, jsonFile, indent=4)
            current_jsonfile  =  os.path.join(exp_folder_ic, "{}.json".format(mydata["name"])) 
            list_oj_jobfiles.append(current_jsonfile)    

with open(os.path.join(outfolder, 'list_of_exp_files.txt'), 'w') as fff1:
    for line in list_oj_jobfiles:
        fff1.write( '{}\n'.format(line ))
with open(os.path.join(outfolder, 'number_of_exp_files.txt'), 'w') as fff2:
    fff2.write( str(len(list_oj_jobfiles)))



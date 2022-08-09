
import os
import json
import photon_mc_atmosphere as atm
import numpy as np
import pandas as pd
import photon_mc_numba_fractional as ph

# zz = np.linspace()


# mdfile = sys.argv[1]  # EXPERIMENT FILES - DOES NOT CHANGE, ALWAYS ARG POS #1
mdfile = os.path.join("exp", "laptop.json")
metadata = json.load(open(mdfile, 'r'))
my_outer_band = '16000_22650'
myfreq = 0

dfatm = atm.init_atmosphere_gfdl_singletimestep(metadata,
                                                myband=my_outer_band,
                                                mygpoint=myfreq)

print(dfatm)

A = np.where(3 < np.array([5, 4, 3, 2, 1 ]))[0]
print(A[-1])
LL = {K:len(A) for K,A in zip(dfatm.keys(), dfatm.values())}
print(LL)

# print(np.shape(dfatm['zz']))
# print(np.shape(dfatm['dz']))

# df = pd.DataFrame(dfatm)

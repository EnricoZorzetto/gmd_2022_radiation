# GFDL preprocessing

Instructions to get everything up and running. 

**1. Clone the repository**

```
git clone https://github.com/chaneyn/GFDL_preprocessing.git
```

**2. Install the included conda environment**

```
conda env create --name HMC -f GFDL_preprocessing/yml/HMC2021.yml
source activate HMC
```

**3. Install missing R packages that are used in Python via rpy2**

```
R
install.packages('soiltexture')
quit()
pip install rpy2
```

**4. Install geospatialtools for GFDL**

```
git clone -b gfdl https://github.com/chaneyn/geospatialtools.git
cd geospatialtools
python setup.py install
cd ..
```

**4. Test on the single grid cell experiment**

```
cd GFDL_preprocessing
python driver.py experiments/chaney2017_global_dev_mar2021.json
```

Edit ```experiments/chaney2017_global_dev_mar2021.json``` to update the ouput directory of: dir, gs_template, lm_template, rn_template. 


**A few things to note...**
This currently only works for a single core setup. It is easy to get the multi core to work as well but it is not immediate. Also there are other GFDL-specific datasets that I need to include before you can create the actual database for the entire globe model. For now, I would just play around with the simple that I am giving you. You can change the boundaries and it will setup a 1x1 arcdegree grid cell over any region over the globe.



**8. Enrico:: How to run the code in Gaea**


Example run for producing grid and dataset for a single grid cell (single process)

```
python driver.py experiments/point1.json
```

Example: parallel jobs for producing grid and dataset for a regional domain (For example, ALP for a domain over the Alps or WNA for Western North America. See details in the experiments folder, and change number of cores based on need / domain size)


```
sbatch gaearun.sh
```


Example run for producing a global dataset:

```
sbatch globalrun.sh
```

For reproducing the analysis in the 2021 Enrico's paper

```
sbatch gmd2021run.sh
```


**9. Notes for the gmd_2022_release:**

Tag gmd_2022_release identifies the version used for the paper.

The simulations in the paper can be reproduced by running first


```
conda activate HMC2021mpi
python experiments/experiments_gmd_2021.py
```
This will create a list of experiment files for later use. Then, all experiments can be
run using the script


```
./gmd2021run.sh
```



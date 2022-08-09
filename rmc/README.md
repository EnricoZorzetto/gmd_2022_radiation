# RMC - Radiation Monte Carlo Code

RMC is a radiation Monte Carlo code which numerically simulate solar shortwave radiation over 
a three dimensional surface.


## Installation and dependencies

Use conda, see yml file. Additional it requires the following 

pip install git+https://github.com/chaneyn/geospatialtools.git



## Quickstart

To run a simple simulation (either single core or multi thread) in a personal PC, run one of the two scripts

```bash
run_photomc_singlecore_laptop.py exp/experiment.json
run_photomc_parallel_laptop.py exp/experiment.json
```

To run on a linux cluster with slurm scheduler, use one of the following scripts (used respectively for Duke and NOAA clusters)
```bash
./run.sh
./gaearun.sh
```

To test the code, the following scripts run some comparison with a dataset from the GFDL AM4 model and with the 6S radiation transfer code (over a flat surface)

```bash
main_photon_test_GFDL_singletimestep.py
main_photon_test_Py6S.py
```

The conda environments to run rmc in a MAC OS and in linux cluster [Gaea] are respectively

```bash
rmc_macos.yml
rmc_gaea.yml
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
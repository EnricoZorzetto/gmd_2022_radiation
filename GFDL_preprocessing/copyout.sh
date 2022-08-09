cd /stor/soteria/hydro/shared/enrico/res_hmc/


 find ./ -type d -exec mkdir -p ../res_hmc_light/{} \;

  find ./ -type f -name "*.pck" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "*.h5" -exec cp {} ../res_hmc_light/{} \;

  find ./ -type f -name "hillslopes.tif" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "tiles.tif" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "dem_latlon.tif" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "demns_ea.tif" -exec cp {} ../res_hmc_light/{} \;
#  find ./ -type f -name "coss_ea.tif" -exec cp {} ../bar/{} \;
  find ./ -type f -name "sinscosa_ea.tif" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "sinssina_ea.tif" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "tcf_ea.tif" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "svf_ea.tif" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "radavelev_ea.tif" -exec cp {} ../res_hmc_light/{} \;
  find ./ -type f -name "radstelev_ea.tif" -exec cp {} ../res_hmc_light/{} \;

#  find ../bar/ -type d -exec rmdir {} \;
  tar -zcvf ../res_hmc_light.tar.gz ../res_hmc_light

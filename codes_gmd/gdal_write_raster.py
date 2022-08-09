

import os
import numpy as np
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt

import geospatialtools.gdal_tools as gdal_tools


def write_raster_WGS84_ezdev(outfile, array, lat, lon, nodata = None):
    """
    Write Numpy array to WGS84 tif file
    :param file:
    :param array:
    :param lat:
    :param lon:
    :return:
    """
    xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
    nrows,ncols = np.shape(array)
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin, xres, 0, ymax, 0, -yres)
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???
    output_raster = gdal.GetDriverByName('GTiff').Create(outfile, ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                 # Anyone know how to specify the
                                                 # IAU2000:49900 Mars encoding?
    output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system
                                                       # to the file
    output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster
    if nodata is not None:
        output_raster.GetRasterBand(1).SetNoDataValue(nodata)
    output_raster.FlushCache()
    return


def write_raster_WGS84(outfile, array, lat, lon, nodata = -9999):
    """
    Given array with dim: LAT (NY)  X  LON (NX)
    LAT 0->Ny, North -> South
    LON 0->Nx, West  -> East
    """
    # Use the function in Nate's package:
    # nrows, ncols = np.shape(array)
    xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
    ny, nx = np.shape(array)
    srs = osr.SpatialReference()  # Establish its coordinate encoding
    srs.ImportFromEPSG(4326) # This one specifies WGS84 lat long.
    proj = srs.ExportToWkt()
    # geotransform=(xmin, xres, 0, ymax, 0, -yres)
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform=(xmin, xres, 0, ymax, 0, -yres)
    metadata = {'nx':nx, 'ny':ny, 'nodata':nodata, 'projection':proj, 'gt':geotransform}
    # metadata = {'nx':ncols, 'ny':nrows, 'nodata':nodata, 'projection':proj, 'gt':geotransform}
    gdal_tools.write_raster(outfile, metadata, array)
    return


def write_ea(infile_ll, outfile_ea, logfile, eares=90, interp = 'average'):
    """
    :param eares:
    :return:
    interp: see gdalwarp documentation.
    It can be: average, bilinear
    """
    eaproj = '+proj=moll +lon_0=%.16f +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs'
    # inp_latlon_tif = '%s/myraster.tif' % outdir
    # outp_ea_tif = '%s/myraster_ea.tif' % outdir
    # log = "%s/myraster_ea.log" % outdir
    if os.path.exists(outfile_ea):
        # print('write_ea:: cleaning up old raster file...')
        # os.system("rm -r {}".format(outp_ea_tif))
        os.system("rm -r {}".format(outfile_ea))
    if os.path.exists(logfile):
        # print('write_ea:: cleaning up old log file...')
        os.system("rm -r {}".format(logfile))
    ll_raster = gdal_tools.read_data(infile_ll)
    minlon = ll_raster.minx
    maxlon = ll_raster.maxx
    # maxlon = lon.max()
    # minlon = lon.min()
    # xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]
    # Reproject the dem
    # dem_ea_tif = '%s/dem_ea.tif' % cdir
    lproj = eaproj % float((maxlon + minlon) / 2) # projection center longitude (min distortion)
    # lproj = eaproj % float((xmax + xmin) / 2) # projection center longitude (min distortion)
    # eares = 900.0 # [m] target resolution
    # print(eares)

    # If we want to crop ans already have boundaries in EA projection:
    # os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (
    #     xmin, ymin, xmax, ymax, eares, eares, lproj, output_latlon_tif, dem_ea_tif, log))

    os.system('gdalwarp -r %s -dstnodata -9999 -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (
        interp, eares, eares, lproj, infile_ll, outfile_ea, logfile))

    # os.system('gdalwarp -r bilinear -dstnodata -9999 -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (
    #     eares, eares, lproj, inp_latlon_tif, outp_ea_tif, log))
    return

# TEST:
# 1) read a latlon raster such as svf
# 2) convert to EA
# 3) check it is consistent with Nate's EA projection

if __name__ == "__main__":

    array = np.array(( (0.1, 0.2, 0.3, 0.4), # DIM: LAT (NY)  X  LON (NX)
                       (0.2, 0.3, 0.4, 0.5),
                       (0.3, 0.4, 0.5, 0.6),
                       (0.4, 0.5, 0.6, 0.7),
                       (0.5, 0.6, 0.7, 0.8) ))
    # My image array
    lat = np.array(( (10.0, 10.0, 10.0, 10.0),
                     ( 9.5,  9.5,  9.5,  9.5),
                     ( 9.0,  9.0,  9.0,  9.0),
                     ( 8.5,  8.5,  8.5,  8.5),
                     ( 8.0,  8.0,  8.0,  8.0) ))

    lon = np.array(( (20.0, 20.5, 21.0, 21.5),
                     (20.0, 20.5, 21.0, 21.5),
                     (20.0, 20.5, 21.0, 21.5),
                     (20.0, 20.5, 21.0, 21.5),
                     (20.0, 20.5, 21.0, 21.5) ))


    outdir = os.path.join('/Users', 'ez6263', 'Documents', 'output_maps')  # PRINCETON LAPTOP
    if not os.path.exists(outdir): os.makedirs(outdir)


    outfile = os.path.join(outdir, 'myraster.tif')
    write_raster_WGS84_ezdev(outfile, array, lat, lon)


    res02 = gdal_tools.read_data(outfile)
    res0 = gdal_tools.read_raster(outfile)

    # read back the file:
    # test the wrapper to Nate's write_raster funxtion
    outfile2 = os.path.join(outdir, 'myraster2.tif')
    write_raster_WGS84(outfile2, array, lat, lon)

    # read back the raster using Nate's codes::
    res2 = gdal_tools.read_data(outfile2)
    res = gdal_tools.read_raster(outfile2)


    print(res02.data)
    print(res0)
    print(res2.data)
    print(res)


    # TRANSFORM TO EQUAL ARA PROJECTION:



    inp_latlon_tif = '%s/myraster.tif' % outdir
    outp_ea_tif = '%s/myraster_ea.tif' % outdir
    log = "%s/myraster_ea.log" % outdir
    eares = 1000
    write_ea(inp_latlon_tif, outp_ea_tif, log, eares = eares)

    inp_latlon_tif = '%s/myraster.tif' % outdir
    res_ll = gdal_tools.read_data(inp_latlon_tif)
    outp_ea_tif = '%s/myraster_ea.tif' % outdir
    res_ea = gdal_tools.read_data(outp_ea_tif)

    print(res_ll.nx, res_ll.ny, res_ll.data)
    print(res_ea.nx, res_ea.ny, res_ea.data)

    plt.figure()
    # plt.imshow(res_ea.data)
    plt.imshow(res_ea.data)
    plt.show()


    maxx = res_ea.maxx
    minx = res_ea.minx
    maxy = res_ea.maxy
    miny = res_ea.miny





    outp_latlon_tif = '%s/myraster_retransf.tif' % outdir
    log2 = "%s/myraster_ll.log" % outdir
    # re-transform ea to latlon and check how similar they are
    # there will be differences to due deformation/cropping and averaging
    # os.system('gdalwarp -r average -dstnodata -9999 -te %.16f %.16f %.16f %.16f -tr %.16f %.16f -t_srs "%s" %s %s >& %s' % (
        # xmin, ymin, xmax, ymax, eares, eares, lproj, output_latlon_tif, dem_ea_tif, log))

    # os.system('gdalwarp -s_srs %s -t_srs "EPSG:4326" %s %s >& %s' % (lproj, outp_ea_tif, outp_latlon_tif, log2))
    os.system('gdalwarp -t_srs EPSG:4326 %s %s >& %s' % (outp_ea_tif, outp_latlon_tif, log2))

# plot a global map with the extent of the three bounding boxes
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.patches as mpatches
# import cartopy.io.img_tiles as cimgt

outfigdir = os.path.join('//', 'home', 'enrico', 'Documents',
                            'dem_datasets', 'outfigdir')

from shapely.geometry.polygon import LinearRing

# bounding boxes   S        N       W        E
BBOX_EastAlps = [46.00,  47.00,  12.00,   13.00, ]
# BBOX_Nepal    = [28.00,  29.00,  84.00,   85.00, ]
BBOX_Nepal    = [29.00,  30.00,  81.00,   82.00, ]
BBOX_Peru     = [-14.00,-13.00, -73.00,  -72.00, ]

BBOXES = {'EastAlps':BBOX_EastAlps,
          'Nepal':BBOX_Nepal,
          'Peru':BBOX_Peru}


def main():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()
    ax.stock_img()
    ax.coastlines()

    DOMAINS = ['EastAlps', 'Peru', 'Nepal']
    for dom in DOMAINS:
        mb = BBOXES[dom]
        # CORNERS                 SW         SE         NE          NW
        lat_corners = np.array([mb[0],     mb[0],     mb[1],     mb[1]])
        lon_corners = np.array([mb[2],     mb[3],     mb[3],     mb[2]])

        # ax.text(mb[3] + 0.5, mb[1] + 0.5, dom, transform=ccrs.Geodetic())

        # (4, 2) NUMPY ARRAY WITH COORDINATES OF VERTICES
        poly_corners = np.zeros((len(lat_corners), 2), np.float64)
        poly_corners[:, 0] = lon_corners
        poly_corners[:, 1] = lat_corners

        poly = mpatches.Polygon(poly_corners,
                                closed=True, ec='r', fill=True, lw=6,
                                fc="yellow", transform=ccrs.Geodetic())
        ax.add_patch(poly)

        # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.savefig(os.path.join(outfigdir, 'global_map.png'), dpi = 300)
    plt.show()

if __name__ == '__main__':
    main()

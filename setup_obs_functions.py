"""

These functions are meant to assist the observer
with finding target lists and preparing DMD
slit patterns with SAMOS. 

Written by Dana Koeppe, Feb. 2022

"""


import numpy as np
import pandas as pd
from astropy.table import Table, Column, QTable
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia 






def submit_gaia_query(coordinates, width=4., height=4., **kwargs):

	"""
	
	Query a region based on given coordinates.  
	Since SAMOS has a 3' x 3' field of view, 
	default box size is 4' x 4'.

	input
		coordinates: region to query (SkyCoords)
		width: width of region to query (arcmin)
		height: height of region to query (arcmin)
		**kwargs for the Gaia query (for instance, verbose=True)
	output
		QTable of the result

	"""

	width = u.Quantity(width, u.arcmin)
	height = u.Quantity(height, u.arcmin)

	result = Gaia.query_object_async(coordinate=coordinates, width=width, height=height, **kwargs)

	pandacat = result.to_pandas()
	pandacat = pandacat.where(np.isnan(pandacat.parallax)==False).dropna(how='all')


	out = QTable.from_pandas(pandacat, index=True)

	return out

	


def write_region_file_from_table(filename, x, y, sys='icrs'):

	"""
	Given some astropy.Table object with columns for RA and DEC,
	write a region file to be read by DS9.
	
	x, y: columns containing x and y positions of points for the regions.
	sys: Coordinate system of the regions.
		Acceptable system names from http://ds9.si.edu/doc/ref/region.html -
			PHYSICAL                # pixel coords of original file using LTM/LTV
			IMAGE                   # pixel coords of current file
			FK4,B1950               # sky coordinate systems
			FK5,J2000               # sky coordinate systems
			ICRS                    # sky coordinate systems
			GALACTIC                # sky coordinate systems
			ECLIPTIC                # sky coordinate systems
			WCS                     # primary WCS
			WCSA                    # secondary WCS
			LINEAR                  # linear primary WCS
	"""

	out = open(filename, "w")

	for i in range(len(x)):
	    out.write('%s\n'%(sys))
	    xi, yi= x[i], y[i]
	    
	    out.write("point(%.7f,%.7f) # point=boxcircle text={%i} color=blue\n" % (xi, yi, i))

	out.close()




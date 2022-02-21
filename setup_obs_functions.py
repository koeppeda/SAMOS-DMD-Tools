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

from astropy import wcs as astropy_wcs





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

	


def write_region_file_from_coords(filename, x, y, sys='icrs'):

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




def create_wcs(pixscale1, pixscale2, pos_angle, crval1, crval2, naxis1=1056, naxis2=1032,
			   crpix1=528., crpix2=516., cunit1='deg', cunit2='deg', 
			   ctype1='RA---TAN', ctype2='DEC--TAN'):
	
	"""
	Create a WCS object based on scale factors and position angle.
	Defaults to give WCS expected for SAMOS Imaging camera.
	Creates a WCS object using the new wcs fits header keywords (CD matrix.). 

	input:
		pixelscale(1,2) : pixel scale in arcsec/pixel for x and y directions.
		pos_angle       : position angle of image.
		crval(1,2)      : RA and DEC reference coordinates.
		naxis(1,2)      : Size of axes for which the WCS will be applied. 
		crpix(1,2)      : Reference pixel coordinates.
		cunit(1,2)      : Units of crvals
		ctype(1,2)      : projection type for crvals.


	"""
	
	CD1_1 = (pixscale1)*np.cos(np.deg2rad(pos_angle))/3600.
	CD1_2 = (pixscale2)*np.sin(np.deg2rad(pos_angle))/3600.
	CD2_1 = -(pixscale1)*np.sin(np.deg2rad(pos_angle))/3600.
	CD2_2 = (pixscale2)*np.cos(np.deg2rad(pos_angle))/3600.

	
	w = astropy_wcs.WCS(header={
		'NAXIS1': naxis1,         # Width of the output fits/image
		'NAXIS2': naxis2,         # Height of the output fits/image
		'WCSAXES': 2,           # Number of coordinate axes
		'CRPIX1': crpix1,        # Pixel coordinate of reference point
		'CRPIX2': crpix2,        # Pixel coordinate of reference point
		'CDELT1': (pixscale1* u.arcsec).to(u.deg).value,    # [deg] Coordinate increment at reference point
		'CDELT2': (pixscale2* u.arcsec).to(u.deg).value,   # [deg] Coordinate increment at reference point
		'CUNIT1': 'deg',        # Units of coordinate increment and value
		'CUNIT2': 'deg',        # Units of coordinate increment and value
		'CTYPE1': 'RA---TAN',   # Right ascension, gnomonic projection
		'CTYPE2': 'DEC--TAN',   # Declination, gnomonic projection 
		'CRVAL1': crval1,    # [deg] Coordinate value at reference point
		'CRVAL2': crval2,   # [deg] Coordinate value at reference point
		'CD1_1' : -CD1_1,        # [deg/pixel] element of scale/rotation matrix is CDELT1*cos(pa) negative beause RA is backwards in SOAR images.
		'CD1_2' : CD1_2,        # [deg/pixel] element of scale/rotation matrix is -CDELT2*sin(pa)
		'CD2_1' : CD2_1,        # [deg/pixel] element of scale/rotation matrix is CDELT1*sin(pa)
		'CD2_2' : CD2_2,        # [deg/pixel] element of scale/rotation matrix is CDELT2*cos(pa)

	})
							   
	return w
   


#"""
class DMDSlit:

	def __init__(self, ra, dec, x, y, dx1=3, dy1=1, dx2=3, dy2=0, slit_n=None):
		
		
		self.ra = ra  # ra [deg] of slit target center
		self.dec = dec # dec [deg] of slit target center
		self.x = x   #ra position in DMD mirrors
		self.y = y   #dec position in DMD mirrors
		self.dx1 = dx1 # left edge  : number of mirrors left of center
		self.dy1 = dy1 # lower edge : number of mirros below center
		self.dx2 = dx2 # right edge : number of mirrors right of center
		self.dy2 = dy2 # upper edge : number of mirrors above center
		#default slit width (y-direction) is 2 mirrors 
		self.slit_n = slit_n #slit number if slit is part of DMD pattern

#"""
#"""


class DMDPattern:



	def __init__(self, Pattern_ID, RA_Center, DEC_Center, Sky_to_DMD_Transform):

		self.Pattern_ID = Pattern_ID # Pattern number ID (because there will prob. 
									 # 		be more than one pattern throughout a night.)
		
		self.RA_Center = RA_Center   # RA center of FoV for this pattern
		self.DEC_Center = DEC_Center # DEC center of FoV for this pattern
		self.Sky_to_DMD_Transform = Sky_to_DMD_Transform
		self.Slit_List = None
		self.Slit_Width_List = None
		#self.Slit_




	def add_slit(self, ra, dec, x, y, dx1=3, dy1=1, dx2=3, dy2=0, slit_n=None):

		slit = DMDSlit(ra=ra, dec=dec, x=x, y=y, dx1=dx1, dy1=dy1, dx2=dx2, dy2=dy2, slit_n=slit_n)

		if self.Slit_List is None:
			self.Slit_List = [slit]

		else:
			self.Slit_List = self.Slit_List.append(slit)
#"""





def create_slit_pattern(target_table, pixel_scale, slit_xsize=7, slit_ysize=3):

	"""
	Given slit sizes/widths, and a list of targets, create a pattern 
	where source spectra won't overlap each other. 

	Input:
		  target_table  - Table with targets.  Must have columns for RA and DEC in J2000d
			pixel_scale - Scale of CCD pixels (arcseconds/pixel)
			 slit_xsize - int or array of ints representing length (in pixels) of slit(s) in x-direction
			 slit_ysize - int or array ints representing the width (in pixels) of slit(s) in y-direction.
		
	"""

	sky2pix = astropy_wcs.utils.skycoord_to_pixel
	
	slit_ra_centers = target_table['ra']

	half_slit_xsize = slit_xsize/2.
	slit_edges_left = slit_ra_centers - (half_slit_xsize*pixel_scale / 3600.) #put into degrees bc that is unit of RA list
	slit_edges_right = slit_ra_centers + (half_slit_xsize*pixel_scale / 3600.) 
	print(slit_edges_left)
	target_table['slit_edges_left'] = slit_edges_left
	target_table['slit_edges_right'] = slit_edges_right

	#center of mass of the targets system
	centerfield = (min(slit_edges_left)+max(slit_edges_left)) / 2.

	#range in pixels of the targets
	range_pixels = (max(slit_edges_left)-min(slit_edges_left))*3600./pixel_scale

	good_index = [0]
	pix_right = []
	pix_left = []
	j=0 

	for i in range(len(target_table)-1):

		if j-i >= 1: 
			continue

		j=i+1
		print(j,i)
		while ((target_table.iloc[j]['slit_edges_left'] < target_table.iloc[i]['slit_edges_right']) & (j<len(target_table)-1)):
			print('skipping target', j)
			j+=1
		print('accepting target', j)
		good_index.append(j)


		pix_left.append( (target_table.iloc[i]['slit_edges_left']-centerfield)*3600./pixel_scale)
		pix_right.append( (target_table.iloc[i]['slit_edges_right']-centerfield)*3600./pixel_scale)


	return good_index, pix_right, pix_left

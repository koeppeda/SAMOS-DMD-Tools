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
from astroquery.vizier import Vizier

from astropy import wcs as astropy_wcs
from astropy.visualization import simple_norm


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, Cursor, TextBox
from matplotlib.patches import Rectangle


import ipywidgets as widgets
from IPython.display import display, HTML

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


def vizier_gaia_query(coordinates, width=4., height=4.,**kwargs):
	# basically for when gaia query is down

	width = u.Quantity(width, u.arcmin)
	height = u.Quantity(height, u.arcmin)


	#want to make sure selected table has certain columns
	viz = Vizier(columns=["RAJ2000", "DEJ2000", "parallax", "BPmag", "Gmag"])
	results_filtered = viz.query_region(coordinates=coordinates, width=width, height=height, catalog='gaia')

	#next 4 lines get the name of the first table in the return list with the max number of columns
	collens = [len(i.columns) for i in results_filtered] 
	max_collen_locs, = np.where(collens==np.max(collens)) 
	max_collen_loc = int(max_collen_locs[0])
	table_name = results_filtered.keys()[max_collen_loc]

	#now redo the query to get the full table, not just the filtered columns
	#I don't know if there is a way to do this in fewer steps with astroquery
	viz_tables = Vizier.query_region(coordinates=coordinates, width=width, height=height, catalog=table_name)
	table = viz_tables[0]

	pandacat = table.to_pandas()#.drop(columns='index')

	pandacat = pandacat.where(np.isnan(pandacat.Plx)==False).dropna(how='all')
	#pandacat = pandacat.rename(columns={"RA_ICRS": "ra", "e_RA_ICRS": "ra_error",
	#                                    "DE_ICRS":"dec", "e_DE_ICRS":"dec_error",
	#                                   "BPmag":"phot_bp_mean_mag", "RPmag":"phot_rp_mean_mag",
	#                                   "Plx": "parallax", "e_Plx":"parallax_error",
	#                                   "Source":"DESIGNATION"})
	
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
	print(CD2_2)

	
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
   


#def download_from_HiPS(hips_)



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


def create_single_dmd_slit_from_target_row(target_row, ra_center, dec_center, wcs=None, slit_xsize=7, slit_ysize=3, pos_angle=0.):

	"""
	Given slit sizes/widths, and a list of targets, create a pattern 
	where source spectra won't overlap each other. 

	Input:
			target_table  - Table with targets.  Must have columns for RA and DEC in J2000d
			wcs 		  - WCS for transforming target coordinates to pixel/mirror  coordinate.  Default 
							is to create WCS for DMD directly.  Otherwise, can go from CCD pixels.
			slit_xsize    - int or array of ints representing length (in mirrors) of slit(s) in x-direction
			slit_ysize    - int or array ints representing the width (in mirrors) of slit(s) in y-direction.
		
	"""



	if wcs is None:
		# or is it 3.1???  
		scale = 3.095*60./1080 # 3.095 arcmin fov projected over 1080x1080 mirrors

		mx_center = 540 #center x mirror of 1080x1080 array
		my_center = 540
		wcs = create_wcs(pixscale1=scale,pixscale2=scale, pos_angle=0., 
						naxis1=1080, naxis2=1080, crval1=ra_center, crval2=dec_center,
						crpix1=mx_center,crpix2=my_center)
	else:
		scale = np.abs(wcs.pixel_scale_matrix[0,0])*3600


	
	from astropy.wcs.utils import skycoord_to_pixel as sky2dmd

	
	slit_ra_center = target_row['ra'].values[0]
	slit_dec_center = target_row['dec'].values[0]


	half_slit_xsize = slit_xsize/2.
	half_slit_ysize = slit_ysize/2. 

	# I am referencing images from Strasburg and Aladin to write this. 
	# Those images are oriented such that RA decreases from left to right, 
	# which is why the input table and the left and right slit eges below 
	# are also sorted in order of decreasing RA.

	slit_edge_left = slit_ra_center + (half_slit_xsize*scale / 3600.) #put into degrees bc that is unit of RA list
	slit_edge_right = slit_ra_center - (half_slit_xsize*scale / 3600.) 

	
	slit_edge_top = slit_dec_center + (half_slit_ysize*scale / 3600.)
	#print("tops",slit_edges_top)
	slit_edge_bottom = slit_dec_center - (half_slit_ysize*scale / 3600.)

	#print(dmd_scale, slit_ra_centers,slit_edges_left)
	target_row['slit_edge_left'] = slit_edge_left
	target_row['slit_edge_right'] = slit_edge_right
	target_row['slit_edge_top'] = slit_edge_top
	target_row['slit_edge_bottom'] = slit_edge_bottom

	#center of mass of the targets system
	centerfield = (min(slit_edge_left)+max(slit_edge_left)) / 2.
	centerfield_dec = (min(slit_edge_bottom)+max(slit_edge_bottom)) / 2.
	#print("center_field ra,dec",centerfield,centerfield_dec)
	#range in mirrors of the targets
	#range_mirrors = (max(slit_edges_left)-min(slit_edges_left))*3600./scale

	
	
	slit_num = 0

	print('creating single target slit')

	ra, dec = target_row.loc[0, ['ra', 'dec']].values
	coords0 = SkyCoord(target_row['ra'].values[0],
					   target_row['dec'].values[0],unit='deg')

	#print(coords0)
	dmd_coords = sky2dmd(coords=coords0, wcs=wcs) # center of target coords in dmd coords.
	dmd_xc, dmd_yc = dmd_coords
	try:
		half_slx = half_slit_xsize[0]
		half_sly = half_slit_ysize[0]
	except:
		half_slx = half_slit_xsize
		half_sly = half_slit_ysize
	
	ml = dmd_xc-half_slx
	mr = dmd_xc+half_slx
	mt = dmd_yc+half_sly
	mb = dmd_yc-half_sly


	mir_left = [ml]
	mir_right = [mr]
	mir_top = [mt]
	mir_bottom = [mb]
	mir_x = [dmd_xc]
	mir_y = [dmd_yc]


	dx1 = np.asarray(mir_x)-np.asarray(mir_left)
	dx2 = np.asarray(mir_right)-np.asarray(mir_x)
	dy1 = np.asarray(mir_top)-np.asarray(mir_y)
	dy2 = np.asarray(mir_y)-np.asarray(mir_bottom)

	sl = DMDSlit(ra=ra,dec=dec, x=dmd_xc, y=dmd_yc, dx1=dx1, dx2=dx2, 
				 dy1=dy1, dy2=dy2, slit_n=slit_num)
		
		
	pcols = ["target", "ra", "dec", "x", "y", "dx1", "dy1", "dx2", "dy2"]
	pdata = np.vstack((target_row['DESIGNATION'].values[0],
					  target_row['ra'].values[0], target_row['dec'].values[0],
					  np.asarray(mir_x), np.asarray(mir_y), dx1, dy1, dx2, dy2)).T
	slit_tab = pd.DataFrame(data=pdata, columns=pcols)


	return slit_tab, sl


def create_dmd_pattern_from_target_table(target_table, wcs=None, ra_center=None, dec_center=None, slit_xsize=7, slit_ysize=3, pos_angle=0.):

	"""
	Given slit sizes/widths, and a list of targets, create a pattern 
	where source spectra won't overlap each other. 

	Input:
			target_table  - Table with targets.  Must have columns for RA and DEC in J2000d
			wcs 		  - WCS for transforming target coordinates to pixel/mirror  coordinate.  Default 
							is to create WCS for DMD directly.  Otherwise, can go from CCD pixels.
			slit_xsize    - int or array of ints representing length (in mirrors) of slit(s) in x-direction
			slit_ysize    - int or array ints representing the width (in mirrors) of slit(s) in y-direction.
		
	"""

	if len(target_table)>1:
		print("length > 1")
		target_table = target_table.sort_values(by="ra", ascending=False)


	if wcs is None:
		# or is it 3.1???  
		scale = 3.095*60./1080 # 3.095 arcmin fov projected over 1080x1080 mirrors

		mx_center = 540 #center x mirror of 1080x1080 array
		my_center = 540
		wcs = create_wcs(pixscale1=scale,pixscale2=scale, pos_angle=0., 
						naxis1=1080, naxis2=1080, crval1=ra_center, crval2=dec_center,
						crpix1=mx_center,crpix2=my_center)
	else:
		scale = np.abs(wcs.pixel_scale_matrix[0,0])*3600

	if (ra_center is None) and (dec_center is None):
		ra_center = (np.max(target_table['ra'])-np.min(target_table['ra'])) /2. + np.min(target_table['ra'])
		dec_center = (np.max(target_table['dec'])-np.min(target_table['dec'])) /2. + np.min(target_table['dec'])
	
	
	
	
	from astropy.wcs.utils import skycoord_to_pixel as sky2dmd

	
	slit_ra_centers = target_table['ra']
	slit_dec_centers = target_table['dec']


	half_slit_xsize = slit_xsize/2.
	half_slit_ysize = slit_ysize/2. 

	# I am referencing images from Strasburg and Aladin to write this. 
	# Those images are oriented such that RA decreases from left to right, 
	# which is why the input table and the left and right slit eges below 
	# are also sorted in order of decreasing RA.

	slit_edges_left = slit_ra_centers + (half_slit_xsize*scale / 3600.) #put into degrees bc that is unit of RA list
	slit_edges_right = slit_ra_centers - (half_slit_xsize*scale / 3600.) 

	
	slit_edges_top = slit_dec_centers + (half_slit_ysize*scale / 3600.)
	#print("tops",slit_edges_top)
	slit_edges_bottom = slit_dec_centers - (half_slit_ysize*scale / 3600.)

	#print(dmd_scale, slit_ra_centers,slit_edges_left)
	target_table['slit_edges_left'] = slit_edges_left
	target_table['slit_edges_right'] = slit_edges_right
	target_table['slit_edges_top'] = slit_edges_top
	target_table['slit_edges_bottom'] = slit_edges_bottom

	#center of mass of the targets system
	centerfield = (min(slit_edges_left)+max(slit_edges_left)) / 2.
	centerfield_dec = (min(slit_edges_bottom)+max(slit_edges_bottom)) / 2.
	#print("center_field ra,dec",centerfield,centerfield_dec)
	#range in mirrors of the targets
	range_mirrors = (max(slit_edges_left)-min(slit_edges_left))*3600./scale

	good_index = []
	mir_x = []
	mir_y = []
	mir_right = []
	mir_left = []
	mir_top = []
	mir_bottom = []
	dmd_slits = []
	j=0 
	slit_num = 0
	for i in range(len(target_table)-1):

		if i==0:
			print('accepting first target', i)
			good_index.append(i)
			ra, dec = target_table.loc[i, ['ra', 'dec']].values
			coords0 = SkyCoord(ra, dec,unit='deg')
			

			#print(coords0)
			dmd_coords = sky2dmd(coords=coords0, wcs=wcs) # center of target coords in dmd coords.
			dmd_xc, dmd_yc = dmd_coords
			try:
				half_slx = half_slit_xsize[i]
				half_sly = half_slit_ysize[i]
			except:
				half_slx = half_slit_xsize
				half_sly = half_slit_ysize
			ml = dmd_xc-half_slx
			mr = dmd_xc+half_slx
			mt = dmd_yc+half_sly
			mb = dmd_yc-half_sly

			print("append slit {}".format(i))
			mir_left.append(ml)
			mir_right.append(mr)
			mir_top.append(mt)
			mir_bottom.append(mb)
			mir_x.append(dmd_xc)
			mir_y.append(dmd_yc)

			dx1 = dmd_xc - ml 
			dx2 = mr - dmd_xc
			dy1 = mt - dmd_yc
			dy2 = dmd_yc - mb

		
			sl = DMDSlit(ra=ra,dec=dec, x=dmd_xc, y=dmd_yc, dx1=dx1, dx2=dx2, 
					 dy1=dy1, dy2=dy2, slit_n=slit_num)
			

			dmd_slits.append(sl)
			slit_num+=1
		
		if j-i >= 1: 
			continue

		j=i+1


		while ((target_table.iloc[j]['slit_edges_left'] > target_table.iloc[i]['slit_edges_right']) & (j<len(target_table)-1)):
								### '>' because images from Strasbourg decrease in RA from left to right.
			print('skipping target', j)
			#print(target_table.iloc[j]['slit_edges_left'], target_table.iloc[i]['slit_edges_right'])

			j+=1
		
		print('accepting target', j)
		print('\n')
		good_index.append(j)


		ra, dec = target_table.loc[j, ['ra', 'dec']].values
		coords0 = SkyCoord(ra, dec,unit='deg')

		dmd_xc, dmd_yc = sky2dmd(coords0, wcs) # center of target coords in dmd coords.

		try:
			half_slx = half_slit_xsize[j]
			half_sly = half_slit_ysize[j]
		except:
			half_slx = half_slit_xsize
			half_sly = half_slit_ysize
		ml = dmd_xc-half_slx
		mr = dmd_xc+half_slx
		mt = dmd_yc+half_sly
		mb = dmd_yc-half_sly

		print("append slit {}".format(j))
		mir_left.append(ml)
		mir_right.append(mr)
		mir_top.append(mt)
		mir_bottom.append(mb)
		mir_x.append(dmd_xc)
		mir_y.append(dmd_yc)
		

		dx1 = dmd_xc - ml 
		dx2 = mr - dmd_xc
		dy1 = mt - dmd_yc
		dy2 = dmd_yc - mb


	
		sl = DMDSlit(ra=ra,dec=dec, x=dmd_xc, y=dmd_yc, dx1=dx1, dx2=dx2, 
				 dy1=dy1, dy2=dy2, slit_n=slit_num)
		



		dmd_slits.append(sl)
		slit_num+=1
		
	dx1 = np.asarray(mir_x)-np.asarray(mir_left)
	dx2 = np.asarray(mir_right)-np.asarray(mir_x)
	dy1 = np.asarray(mir_top)-np.asarray(mir_y)
	dy2 = np.asarray(mir_y)-np.asarray(mir_bottom)

	pcols = ["target", "ra", "dec", "x", "y", "dx1", "dy1", "dx2", "dy2"]
	pdata = np.vstack((target_table.loc[good_index, 'DESIGNATION'].values,
					  target_table.loc[good_index, 'ra'].values, target_table.loc[good_index, 'dec'].values,
					  np.asarray(mir_x), np.asarray(mir_y), dx1, dy1, dx2, dy2)).T


	pattern_table = pd.DataFrame(data=pdata, columns=pcols)

	return pattern_table, dmd_slits, good_index


def DMDPatternsMaker(target_table, wcs, ra_center, dec_center, pos_angle=0.):

	need_patterns = target_table.copy()

	patterns = []
	slits_lists = []
	ind_lists = []
	while len(need_patterns)>1:

		print("creating DMD pattern {}".format(len(patterns)+1))


		

		pattern, slits, inds = create_dmd_pattern_from_target_table(need_patterns, wcs=wcs, 
																	ra_center=ra_center, dec_center=dec_center, 
																	slit_xsize=need_patterns.dx.values, 
																	slit_ysize=need_patterns.dy.values, 
																	pos_angle=pos_angle)

		patterns.append(pattern)
		slits_lists.append(slits)
		ind_lists.append(inds)
		need_patterns = need_patterns.drop(labels=inds, axis='index').reset_index(drop=True)


		print("pattern table made {} to go".format(len(need_patterns)))

		

	if len(need_patterns)==1:

		print("last pattern is a single slit")
		
		
		
		needs_pattern = pd.DataFrame(data=[need_patterns.iloc[0]], columns=need_patterns.columns)
		
		pattern_tab, slit = create_single_dmd_slit_from_target_row(needs_pattern, wcs=wcs, 
																ra_center=ra_center, dec_center=dec_center, 
																slit_xsize=needs_pattern.dx.values, 
																slit_ysize=needs_pattern.dy.values, 
																pos_angle=pos_angle)

		patterns.append(pattern_tab)
		slits_lists.append([slit])
		
		
	print("{} DMD patterns created".format(len(patterns)))
	return patterns, slits_lists



from matplotlib .gridspec import GridSpec

class JupyterSlitSizeIterator:
	
	#def __init__(self, df, fig, ax, image, ind=0, xs=7, ys=3):
	def __init__(self, df, image, ind=0, xs=7, ys=3):	
		
		df = df.sort_values(by='ra', ascending=False).reset_index(drop=True) # make sure index iteration is easy
		self.df = df # table of gaia targets
		self.imshape = image.shape
		


		##self.fig = fig
		#self.ax1 = ax
		#self.ax2 = self.fig.add_subplot(122)
		#self.fig.subplots_adjust(hspace=10)
		#figw, figh = fig.get_size_inches()
		#self.fig.set(figwidth=1.8*figw)

		self.fig = plt.figure(figsize=(10,7.5), constrained_layout=False)
		self.gs = GridSpec(figure=self.fig, nrows=1, ncols=2, wspace=0.05)
		self.ax1 = self.fig.add_subplot(self.gs[0], )
		self.ax2 = self.fig.add_subplot(self.gs[1])#, anchor=(0.5,0))

		self.fig.subplots_adjust(bottom=0.3, left=0.08)
		norm = simple_norm(data=image, stretch='log')
		self.im1 = self.ax1.imshow(image, origin='lower',cmap='gray',norm=norm,)#extent=[-512, 512, -512, 512])
		self.ax1.set_title("zoomed in view")
		self.ax1.set_xlabel("NAXIS1", fontsize=12)
		self.ax1.set_ylabel("NAXIS2", fontsize=12)
		self.im2 = self.ax2.imshow(image, origin='lower',cmap='gray', norm=norm)
		self.ax2.tick_params(labelright=True, right=True, labelleft=False, left=False)
		self.ax2.set_title("full FoV")
		self.ax2.set_xlabel("NAXIS1", fontsize=12)
		self.ax2.set_ylabel("NAXIS2", fontsize=12)
		self.ax2.yaxis.set_label_position('right')
		self.xs = xs
		self.ys = ys
		self.sl_xsizes = []
		self.sl_ysizes = []
		self.ind = ind # start index of df for slit iteration


		self.df_output = widgets.Output()

		sdf_cols = ['slit_num','ra', 'dec', 'x', 'y', 'dx', 'dy']
		self.slits_df = pd.DataFrame(columns=sdf_cols)

		with self.df_output:
			display(HTML(self.slits_df.to_html(notebook=True,border=1,justify='center')))


	def __call__(self):

		targ1 = self.df.iloc[self.ind]
		tbox1 = Rectangle(xy=(targ1.x-(self.xs/2.),targ1.y-(self.ys/2.)),
							width=self.xs, height=self.ys)
		
		self.tbox1 = tbox1 # target box

		#tbox2 = Rectangle(xy=(targ1.x-(xs/2.),targ1.y-(ys/2.)),
		#					width=xs, height=ys)
		

		self.ax1.add_patch(self.tbox1)
		
		titletxt = "{} of {} targets".format(self.ind+1,self.df.shape[0])
		self.fig.suptitle(titletxt,va='center', y=0.9, fontsize=20)
		
		xlim,ylim = self.get_new_axlims(targ1)
		self.ax1.set_xlim(xlim)
		self.ax1.set_ylim(ylim)

		tbox2w = xlim[1]-xlim[0]
		tbox2h = ylim[1]-ylim[0]
		tbox2 = Rectangle(xy=(xlim[0],ylim[0]), width=tbox2w, height=tbox2h, 
			edgecolor="red", linewidth=2, fill=False)
		self.tbox2 = tbox2 # target box
		self.ax2.add_patch(self.tbox2)
		
		self.targ = targ1
		

		#axcoords = plt.axes(figure=self.fig, position=[0.18, 0.22, 0.62, 0.04])
		self.figcoordtxt = "RA={}, DEC={}"
		axcoords_s = self.figcoordtxt.format(self.targ.ra, self.targ.dec)
		self.figcoord_txtbox = plt.text(x=-700,y=-200,s=axcoords_s,fontsize=13,figure=self.fig)


		axsize = plt.axes(figure=self.fig,position=[0.18, 0.18, 0.62, 0.04])
		self.x_slider = Slider(
			ax=axsize,
			label=r"slit size $\Delta$X mirrors",
			valmin=0,
			valmax=100,
			valinit=7,
			valstep=1,
			orientation="horizontal"
		)

		aysize = plt.axes(figure=self.fig,position=[0.18, 0.14, 0.62, 0.04])
		self.y_slider = Slider(
			ax=aysize,
			label=r"slit size $\Delta$Y mirrors",
			valmin=0,
			valmax=100,
			valinit=3,
			valstep=1,
			orientation="horizontal"
		)
		
		self.x_slider.on_changed(self.update)
		self.y_slider.on_changed(self.update)

		#axinput = plt.axes(figure=self.fig, position=[0.81, 0.25, 0.09, 0.042])
		#ayinput = plt.axes(figure=self.fig, position=[0.81, 0.21, 0.09, 0.042])
		#self.xs_input = TextBox(ax=axinput, initial=xs, label=r"Enter $\Delta$X", textalignment="right", label_pad=-1.8)
		#self.ys_input = TextBox(ax=ayinput, initial=ys, label=r"Enter $\Delta$Y", textalignment="right", label_pad=-1.8)
		#self.xs_input.on_submit(self.set_xsize)
		#self.ys_input.on_submit(self.set_ysize)
		
		
		# Create a `matplotlib.widgets.Button` to 
		# go to next target and reset sliders to initial values.
		nextax = plt.axes([0.78, 0.08, 0.1, 0.04])
		nbutton = Button(nextax, 'Next', hovercolor='0.975')
		self.nbutton = nbutton
		self.nbutton.on_clicked(self.next_targ)
		
		prevax = plt.axes([0.08, 0.08, 0.1, 0.04])
		pbutton = Button(prevax, 'Prev', hovercolor='0.975')
		self.pbutton = pbutton
		self.pbutton.on_clicked(self.prev_targ)
		

		acceptbuttonax = plt.axes([0.38, 0.08, 0.2, 0.04])
		acceptbutton = Button(acceptbuttonax, 'accept slit', hovercolor='magenta')
		self.acceptbutton = acceptbutton
		self.acceptbutton.on_clicked(self.accept_slit)


		tdata = [self.targ.ra.round(8),self.targ.dec.round(8), self.targ.x.round(3), 
				self.targ.y.round(3), self.x_slider.val, self.y_slider.val]

		"""
		# this is a backup table method, but it sucks

		sltable_txt = ["{}".format(val) for val in tdata]
		row_ind_labels = ["slit {:03n}".format(i+1) for i in range(len(self.sl_xsizes))]
		row_ind_labels.extend(["slit {:03n}".format(self.ind+1)])
		self.cell_texts = [sltable_txt]
		#print(len(tdata), sltable_txt, len(row_ind_labels))
		slit_table = plt.table(cellText=self.cell_texts, bbox=(-1.2, -1.8, 3.5, 1.5), figure=self.fig,
								in_layout=True, rowLabels = row_ind_labels,
								colLabels=["RA", "DEC", "X", "Y", "dX", "dY"],
								loc='bottom',colWidths=[0.05, 0.05, 0.03, 0.03, 0.01, 0.01])
		#slit_table.auto_set_column_width(False)
		slit_table.auto_set_font_size(False)
		slit_table.set_fontsize(12)
		self.slit_table = slit_table
		"""
	
	def get_new_axlims(self, targ):

		xlim = max(targ.x-100,0),min(targ.x+100, self.imshape[0])
		ylim = max(targ.y-100,0),min(targ.y+100, self.imshape[1])
		
		return xlim, ylim


	def set_xsize(self, val):

		self.x_slider.set_val(val)

	def set_ysize(self, val):

		self.y_slider.set_val(val)


	def adjust_axis_positions(self, ax):

		box = ax.get_position()
		newbox = box

		box_dy = box.y1-box.y0

		newbox.y0+=0.15
		newbox.y1 = newbox.y0+box_dy 

		return newbox

	# The function to be called anytime a slider's value changes
	def update(self, event):

		centerx, centery = self.targ.x, self.targ.y 
		
		patchx = centerx-(self.x_slider.val/2.)
		patchy = centery-(self.y_slider.val/2.)
		
		#print("update")
		
		self.tbox1.set_x(patchx)
		self.tbox1.set_y(patchy)
		self.tbox1.set_width(self.x_slider.val)
		self.tbox1.set_height(self.y_slider.val)


		self.tbox1.figure.canvas.draw()
		
		#self.fig.figure.canvas.draw()



	
	# The function called any time you go to a new target in the list
	def update_patch(self):#, targ):

		#self.x_slider.reset()
		#self.y_slider.reset()
		targ=self.targ
		patchx,patchy = self.get_targ_patch_cornerxy(targ)

		self.tbox1.set_x(patchx)
		self.tbox1.set_y(patchy)
		self.tbox1.set_width(7)#(self.x_slider.val)
		self.tbox1.set_height(3)#(self.y_slider.val)

		xlim,ylim = self.get_new_axlims(targ)
		self.ax1.set_xlim(xlim)
		self.ax1.set_ylim(ylim)

		
		tbox2w = xlim[1]-xlim[0]
		tbox2h = ylim[1]-ylim[0]
		

		self.tbox2.set_x(xlim[0])
		self.tbox2.set_y(ylim[0])
		self.tbox2.set_width(tbox2w)
		self.tbox2.set_height(tbox2h)

		axcoords_s = self.figcoordtxt.format(self.targ.ra, self.targ.dec)
		self.figcoord_txtbox.set_text(axcoords_s)

		#test_txt = r"next target button clicked. self.ind={}, targ.ind={}".format(self.ind, targ.name)
		#try:
		#	self.button_txt.set(text=test_txt)
		#except:
		#	self.button_txt = self.ax1.annotate(xy=(150,15), text=test_txt,fontsize=16, xycoords='figure points')


		self.tbox1.figure.canvas.draw()
		self.tbox2.figure.canvas.draw()
		self.fig.figure.canvas.draw()

	def get_targ_patch_cornerxy(self, targ):

		#targ = self.df.iloc[ind]
		
		xc = targ.x
		yc = targ.y
		
		patchx = xc-(self.x_slider.val/2.)
		patchy = yc-(self.y_slider.val/2.)

		return patchx, patchy

	def accept_slit(self, event):

		sdf_cols = ['slit_num','ra', 'dec', 'x', 'y', 'dx', 'dy']
		slit_num_txt = "{:03n}".format(self.ind+1)
		slit_row = [np.hstack((slit_num_txt, self.df.loc[self.ind,['ra', 'dec', 'x', 'y']].values, self.x_slider.val, self.y_slider.val))]
		slit_df = pd.DataFrame(data=slit_row,columns=sdf_cols,index=[self.ind])

		if self.ind in self.slits_df.index.values:

			self.slits_df.update(slit_df)

			# these lists are obsolete now with the dataframe so idk why it's still here right now
			self.sl_xsizes.pop(self.ind)
			self.sl_xsizes.insert(self.ind,self.x_slider.val)
			self.sl_ysizes.pop(self.ind)
			self.sl_ysizes.insert(self.ind,self.y_slider.val)

		elif self.slits_df.shape[0]<self.df.shape[0]:
			
			self.slits_df = pd.concat((self.slits_df, slit_df), axis=0, ignore_index=False)
			self.sl_xsizes.append(self.x_slider.val)
			self.sl_ysizes.append(self.y_slider.val)

		
		
		with self.df_output:
			self.df_output.clear_output()
			display(HTML(self.slits_df.sort_index(axis='index').to_html(notebook=True,border=2,justify='center')))


		if self.ind>=self.df.shape[0]-1:

			self.df['dx'] = self.slits_df.dx.values#np.asarray(self.sl_xsizes)
			self.df['dy'] = self.slits_df.dy.values#np.asarray(self.sl_ysizes)

			self.ax1.annotate(xy=(100,15), text=r"All slit sizes added to table.",fontsize=16, xycoords='figure points')

			return

		self.next_targ(event)


	def next_targ(self, event):
		print('click', event)
		


		ind = self.ind + 1


		targ = self.df.iloc[ind]
		#self.targ = targ
		#
		
		titletxt = "{} of {} targets".format(ind+1,self.df.shape[0])
		self.fig.suptitle(titletxt, va='center', y=0.9, fontsize=20)

		
		self.targ = targ
		self.ind = ind
		self.update_patch()


		self.x_slider.reset()
		self.y_slider.reset()
		#self.targ = targ
		#self.ind = ind


	def prev_targ(self, event):
		print('click', event)
		
		self.sl_xsizes.append(self.x_slider.val)
		self.sl_ysizes.append(self.y_slider.val)
		

		if self.ind==0:
			print("beginning of list")
			return 


		ind = self.ind - 1

		
		targ = self.df.iloc[ind]
		

		titletxt = "{} of {} targets".format(ind+1,self.df.shape[0])
		self.fig.suptitle(titletxt, va='center', y=0.9, fontsize=20)

		self.targ = targ
		self.ind = ind
		self.update_patch()
		
		self.x_slider.reset()
		self.y_slider.reset()

		#self.tbox1.figure.canvas.draw()
		#self.fig.figure.canvas.draw()
		
		
	

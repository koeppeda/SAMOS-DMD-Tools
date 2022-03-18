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


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, Cursor, TextBox
from matplotlib.patches import Rectangle




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
			coords0 = SkyCoord(target_table.loc[i, 'ra'],
							   target_table.loc[i, 'dec'],unit='deg')

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


			mir_left.append(ml)
			mir_right.append(mr)
			mir_top.append(mt)
			mir_bottom.append(mb)
			mir_x.append(dmd_xc)
			mir_y.append(dmd_yc)


		
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

		mir_left.append(ml)
		mir_right.append(mr)
		mir_top.append(mt)
		mir_bottom.append(mb)
		mir_x.append(dmd_xc)
		mir_y.append(dmd_yc)
		
		sl = DMDSlit(ra=ra,dec=dec, x=dmd_xc, y=dmd_yc, dx1=half_slit_xsize, dx2=half_slit_ysize, slit_n=slit_num)
		dmd_slits.append(sl)
		slit_num+=1
		
		
	pcols = ["target", "ra", "dec", "x", "y", "dx1", "dy1", "dx2", "dy2"]
	pdata = np.vstack((target_table.loc[good_index, 'DESIGNATION'].values,
					  target_table.loc[good_index, 'ra'].values, target_table.loc[good_index, 'dec'].values,
					  np.asarray(mir_x), np.asarray(mir_y), np.asarray(mir_x)-np.asarray(mir_left), 
					  np.asarray(mir_top)-np.asarray(mir_y), np.asarray(mir_right)-np.asarray(mir_x), 
					  np.asarray(mir_y)-np.asarray(mir_bottom))).T
	pattern_table = pd.DataFrame(data=pdata, columns=pcols)

	return pattern_table, dmd_slits, good_index




	import ipywidgets as widgets 




class JupyterSlitSizeIterator:
	
	def __init__(self, df, fig, ax, imshape, ind=0, xs=7, ys=3):
		
		df = df.sort_values(by='ra', ascending=False).reset_index(drop=True) # make sure index iteration is easy
		self.df = df # table of gaia targets
		self.fig = fig
		self.fig.subplots_adjust(left=0., bottom=0.25)

		self.ax = ax
		self.xs = xs
		self.ys = ys
		self.sl_xsizes = []
		self.sl_ysizes = []
		self.ind = ind # start index of df for slit iteration
		targ1 = self.df.iloc[ind]
		tbox = Rectangle(xy=(targ1.x-(xs/2.),targ1.y-(ys/2.)),
							width=xs, height=ys)
		
		self.tbox = tbox # target box

		self.ax.add_patch(self.tbox)
		titletxt = "{} of {} targets".format(self.ind+1,df.shape[0])
		self.ax.set_title(titletxt)
		self.imshape = imshape
		xlim = max(targ1.x-250,0),min(targ1.x+250, imshape[0])
		ylim = max(targ1.y-250,0),min(targ1.y+250, imshape[1])
		self.ax.set_xlim(xlim)
		self.ax.set_ylim(ylim)
		
		axsize = plt.axes(figure=self.fig,position=[0.15, 0.155, 0.62, 0.04])
		self.x_slider = Slider(
			ax=axsize,
			label=r"slit size $\Delta$X",
			valmin=0,
			valmax=100,
			valinit=7,
			valstep=1,
			orientation="horizontal"
		)

		aysize = plt.axes(figure=self.fig,position=[0.15, 0.11, 0.62, 0.04])
		self.y_slider = Slider(
			ax=aysize,
			label=r"slit size $\Delta$Y",
			valmin=0,
			valmax=100,
			valinit=3,
			valstep=1,
			orientation="horizontal"
		)
		
		axinput = plt.axes(figure=self.fig, position=[0.81, 0.155, 0.09, 0.042])
		ayinput = plt.axes(figure=self.fig, position=[0.81, 0.11, 0.09, 0.042])

		self.xs_input = TextBox(ax=axinput, label=r"Enter $\Delta$X", textalignment="right", label_pad=-2.1)
		self.yx_input = TextBox(ax=ayinput, label=r"Enter $\Delta$Y", textalignment="right", label_pad=-2.1)
		self.x_slider.on_changed(self.update)
		self.y_slider.on_changed(self.update)
		
		# Create a `matplotlib.widgets.Button` to 
		# go to next target and reset sliders to initial values.
		nextax = plt.axes([0.75, 0.025, 0.1, 0.04])
		nbutton = Button(nextax, 'Next', hovercolor='0.975')
		self.nbutton = nbutton
		self.nbutton.on_clicked(self.next_targ)
		
		prevtax = plt.axes([0.05, 0.025, 0.1, 0.04])
		pbutton = Button(prevtax, 'Prev', hovercolor='0.975')
		self.pbutton = pbutton
		self.pbutton.on_clicked(self.prev_targ)
		

	
	def get_new_axlims(self, targ):

		xlim = max(targ.x-250,0),min(targ.x+250, self.imshape[0])
		ylim = max(targ.y-250,0),min(targ.y+250, self.imshape[1])
		
		return xlim, ylim

	def get_patch_centerxy(self):

		x,y = self.tbox.get_x(), self.tbox.get_y()

		half_xsize = self.tbox.get_width()/2.
		half_ysize = self.tbox.get_height()/2.
		centerx = x+half_xsize
		centery = y+half_ysize

		return centerx, centery


	# The function to be called anytime a slider's value changes
	def update(self, event):

		centerx, centery = self.get_patch_centerxy()
		
		patchx = centerx-(self.x_slider.val/2.)
		patchy = centery-(self.y_slider.val/2.)
		
		#print("update")
		
		self.tbox.set_x(patchx)
		self.tbox.set_y(patchy)
		self.tbox.set_width(self.x_slider.val)
		self.tbox.set_height(self.y_slider.val)
		self.tbox.figure.canvas.draw()
		self.fig.figure.canvas.draw()


	def get_targ_patch_cornerxy(self, targ):

		#targ = self.df.iloc[ind]
		
		xc = targ.x
		yc = targ.y
		
		patchx = xc-(self.x_slider.val/2.)
		patchy = yc-(self.y_slider.val/2.)

		return patchx, patchy

	def next_targ(self, event):
		print('click', event)
		

		if len(self.sl_xsizes)<self.df.shape[0]:
			self.sl_xsizes.append(self.x_slider.val)
			self.sl_ysizes.append(self.y_slider.val)
		

		if self.ind>=self.df.shape[0]-1:

			self.df['dx'] = np.asarray(self.sl_xsizes)
			self.df['dy'] = np.asarray(self.sl_ysizes)
			self.ax.annotate(xy=(100,15), text=r"All slit sizes added to table.",fontsize=16, xycoords='figure points')

			#self.ax.text(x=0.8,y=0.6, s="Slit sizes added to table.")
			#self.fig.figure.canvas.draw()
			return 


		ind = self.ind + 1

		self.x_slider.reset()
		self.y_slider.reset()

		targ = self.df.iloc[ind]

		patchx,patchy = self.get_targ_patch_cornerxy(targ)

		self.tbox.set_x(patchx)
		self.tbox.set_y(patchy)
		self.tbox.set_width(self.x_slider.val)
		self.tbox.set_height(self.y_slider.val)

		self.ind = ind
		
		xlim,ylim = self.get_new_axlims(targ)
		self.ax.set_xlim(xlim)
		self.ax.set_ylim(ylim)


		titletxt = "{} of {} targets".format(self.ind+1,self.df.shape[0])
		self.ax.set_title(titletxt)

		self.tbox.figure.canvas.draw() # Does this actually do anything?
		self.fig.figure.canvas.draw() # Does this actually do anything?
		
		
	def prev_targ(self, event):
		print('click', event)
		
		self.sl_xsizes.append(self.x_slider.val)
		self.sl_ysizes.append(self.y_slider.val)
		

		if self.ind==0:
			print("beginning of list")
			return 


		ind = self.ind - 1

		self.x_slider.reset()
		self.y_slider.reset()

		targ = self.df.iloc[ind]
		
		patchx, patchy = self.get_targ_patch_cornerxy(targ)

		self.tbox.set_x(patchx)
		self.tbox.set_y(patchy)
		self.tbox.set_width(self.x_slider.val)
		self.tbox.set_height(self.y_slider.val)
		
		
		self.ind = ind
		
		xlim,ylim = self.get_new_axlims(targ)
		self.ax.set_xlim(xlim)
		self.ax.set_ylim(ylim)

		titletxt = "{} of {} targets".format(self.ind+1,self.df.shape[0])
		self.ax.set_title(titletxt)

		self.tbox.figure.canvas.draw()
		self.fig.figure.canvas.draw()
		
		
	

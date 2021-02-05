'''Module for downloading SkyMapper data (image cutouts and catalogs).'''

import numpy as np
from astropy.table import Table, join
from astropy.io import ascii
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import votable
#import requests
try:
   import reproject
except:
   reproject = None


def getImages(ra, dec, size=0.125, filt='H', verbose=False):
   '''Query the 2MASS data server to get a list of images for the given 
      coordinates, size and filter.

      Args:
         ra,dec (float):  RA/DEC in decimal degrees
         size (float):  Size of region in degrees
         filt (str): the filter you want
         verbose(bool): give extra info

      Returns:
         list of filenames
   '''
   templ = "https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?"\
           "POS={},{}&SIZE={}&BAND={}&FORMAT=image/fits"
   baseurl = templ.format(ra, dec, size, filt)
   if verbose: print("About to query: " + baseurl)
   vot = votable.parse(baseurl)
   tab = vot.get_first_table()

   urls = [x[1].decode('utf-8') for x in tab.array]

   return urls
   
def getFITS(ra, dec, size, filters, mosaic=False):
   '''Retrieve the FITS files from SkyMapper server, centered on ra,dec
   and with given size.

   Args:
      ra (float):  RA in degrees
      dec (float):  DEC in degrees
      size (float):  size of FOV in degrees
      filters (str):  filters to get:  e.g gri
      mosaic(bool): If more than one PS images is needed to tile the field,
                    do we mosaic them? Requires reproject module if True

   Returns:
      list of FITS instances
   '''
   if mosaic and reproject is None:
      raise ValueError("To use mosaic, you need to install reproject")
   # max size form image server... need to work on more mosaicking later.
   if size < 0.05:  size = 0.05
   if size > 7: size = 7
   filters = list(filters)
   ret = []
   for filt in filters:
      urls = getImages(ra, dec, size, filt, verbose=True)
      if len(urls) < 1:
         return None
      if len(urls) > 1 and mosaic:
         from reproject import reproject_interp
         from reproject.mosaicking import find_optimal_celestial_wcs
         from reproject.mosaicking import reproject_and_coadd
         fts = [fits.open(url) for url in urls]
         wcs_out,shape_out = find_optimal_celestial_wcs([ft[0] for ft in fts])
         ar_out,footprint = reproject_and_coadd([ft[0] for ft in fts],
             wcs_out, shape_out=shape_out, reproject_function=reproject_interp,
             match_background=True)
         h_out = fts[0][0].header.copy()
         for key in list(h_out.keys()):
            if key in ['CD1_1','CD2_2','CD1_2','CD2_1'] or key[:2] == 'PV':
               h_out.remove(key)
         h_out.update(wcs_out.to_header())

         # Now figure out the subset of data
         wcs = WCS(h_out)
         x0 = ra - size/2/np.cos(dec*np.pi/180)
         x1 = ra + size/2/np.cos(dec*np.pi/180)
         y0 = dec-size/2
         y1 = dec+size/2
         i0,j0 = wcs.wcs_world2pix(x0, y0,0)
         i1,j1 = wcs.wcs_world2pix(x1, y1,0)
         if i1 < i0: i0,i1 = i1,i0
         if j1 < j0: j0,j1 = j1,j0
         print(i0,i1,j0,j1)
         ar_out = ar_out[int(j0):int(j1),int(i0):int(i1)]
         newhdu = fits.PrimaryHDU(ar_out, header=h_out)
         ret.append(fits.HDUList([newhdu]))
      else:
         ret.append(fits.open(urls[0]))

   return ret


def getStarCat(ra, dec, radius):
   '''Get a list of SM stars plus their photometry.'''

   templ = "http://skymapper.anu.edu.au/sm-cone/public/query?RA={}&DEC={}"\
          "&SR={}&VERB=3&RESPONSEFORMAT=CSV&CATALOG=dr1.fs_photometry"
   url = templ.format(ra, dec, radius)

   tab = ascii.read(url)
   if len(tab) == 0:
      return None
   
   tab = tab[~tab['object_id'].mask]
   tab = tab['object_id', 'filter', 'ra_img','decl_img','mag_psf','e_mag_psf']
   tab.rename_column('object_id','objID')
   tab.rename_column('ra_img', 'RA')
   tab.rename_column('decl_img', 'DEC')
   tabg = tab[tab['filter'] == 'g']
   tabr = tab[tab['filter'] == 'r']
   tabi = tab[tab['filter'] == 'i']
   tabg.rename_column('mag_psf','gmag')
   tabg.rename_column('e_mag_psf','gerr')
   tabg.remove_column('filter')
   tabr.rename_column('mag_psf','rmag')
   tabr.rename_column('e_mag_psf','rerr')
   tabr.remove_column('filter')
   tabi.rename_column('mag_psf','imag')
   tabi.rename_column('e_mag_psf','ierr')
   tabi.remove_column('filter')

   newtab = join(tabg, tabr, keys='objID')
   newtab = join(newtab, tabi, keys='objID')
   newtab = newtab['objID','RA','DEC','gmag','gerr','rmag','rerr','imag','ierr']
   gids = np.greater(newtab['gmag'], 0)*np.less(newtab['gmag'],20)
   gids = gids*np.greater(newtab['rmag'], 0)*np.less(newtab['rmag'],20)
   gids = gids*np.greater(newtab['imag'], 0)*np.less(newtab['imag'],20)

   newtab = newtab[gids]
   return newtab


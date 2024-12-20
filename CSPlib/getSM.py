'''Module for downloading SkyMapper data (image cutouts and catalogs).'''

import numpy as np
from astropy.table import Table, join
from astropy.io import ascii
from astropy.io import fits
from astropy.wcs import WCS
#import requests
try:
   import reproject
except:
   reproject = None

# Scale of PS images
SMscale = 0.496/3600   # in degrees/pixel



def getImages(ra, dec, size=0.125, filt='g', verbose=False):
   '''Query the SM data server to get a list of images for the given 
      coordinates, size and filter. We (maybe) check the corners to see if
      we need to have more than one filename.

      Args:
         ra,dec (float):  RA/DEC in decimal degrees
         size (int):  Size of cutout in pixels
         filt (str): the filter you want
         verbose(bool): give extra info

      Returns:
         list of filenames
   '''
   # First, we look to see if we can find images that cover the area
   # entirely and have image_type='main'
   templ = "https://api.skymapper.nci.org.au/public/siap/dr4/query?"\
           "POS={},{}&SIZE={}&BAND={}&FORMAT=image/fits&VERB=3&"\
           "RESPONSEFORMAT=CSV&INTERSECT={}"
   baseurl = templ.format(ra, dec, size, filt, 'COVERS')
   if verbose: print("About to query: " + baseurl)
   for i in range(3):
      try:
         table = Table.read(baseurl, format='ascii')
         success = True
      except:
         success = False
      if success: break
   if not success:
      return []


   gids = table['image_type'] == 'main'
   if np.any(gids):
      # Pick out the deepest image
      idx = np.argmax(table['zpapprox'])
      return [table[idx]['get_fits']]

   # Next, we look for images that contain the SN position
   baseurl = templ.format(ra, dec, size, filt, 'CENTER')
   for i in range(3):
      try:
         table = Table.read(baseurl, format='ascii')
         success = True
      except:
         success = False
      if success: break

   if not success:
      # Not able to contact
      return []
      
   gids = table['image_type'] == 'main'
   if np.any(gids):
      return list(table[gids]['get_fits'])

   # Ugh, gotta settle for shorts I guess
   if len(table) > 0:
      return list(table['get_fits'])

   # So if we get here, we have no coverage
   return []
   
def getFITS(ra, dec, size, filters, mosaic=False, retain=False, maxtries=3):
   '''Retrieve the FITS files from SkyMapper server, centered on ra,dec
   and with given size.

   Args:
      ra (float):  RA in degrees
      dec (float):  DEC in degrees
      size (float):  size of FOV in degrees
      filters (str):  filters to get:  e.g gri
      mosaic(bool): If more than one PS images is needed to tile the field,
                    do we mosaic them? Requires reproject module if True
      retain(bool): If True, keep the individual FITS files 
      maxtries (int): Maximum number of retries to download data

   Returns:
      list of 2-tupes of FITS instances one element for each filter.
      The first of the 2-tuples is the mosaic, the 2nd is the single frame
      with the largest overlap with the requested position. We do this 
      because SkyMpapper doesn't calibrate their images to a common zero-
      point, so a single image is better for template subtractions.
   '''
   if mosaic and reproject is None:
      raise ValueError("To use mosaic, you need to install reproject")
   # max size form image server... need to work on more mosaicking later.
   if size > 0.17: size = 0.17
   filters = list(filters)
   ret = []
   for filt in filters:
      urls = getImages(ra, dec, size, filt)
      if len(urls) < 1:
         return None
      if len(urls) > 1 and mosaic:
         from reproject import reproject_interp
         from reproject.mosaicking import find_optimal_celestial_wcs
         from reproject.mosaicking import reproject_and_coadd
         
         fts = []
         for url in urls:
            tries = 0
            while (tries < maxtries):
               try:
                  newfts = fits.open(url)
                  fts.append(newfts)
                  break
               except:
                  tries += 1
         if len(fts) != len(urls):
            # Something went wrong
            return None

         # Now we get the image with center closest to our position. Easy
         # way is to find image with largest minimum distance to a corner
         wcss = [WCS(ft[0]) for ft in fts]
         feet = [w.calc_footprint() for w in wcss]
         mindists2 = [(((f[:,0]-ra)*np.cos(dec*np.pi/180))**2+(f[:,1]-dec)**2).min() \
               for f in feet]
         idx = np.argmax(mindists2)
         if retain:
            [fts[i].writeto('temp{}.fits'.format(i)) for i in range(len(fts))]
         wcs_out,shape_out = find_optimal_celestial_wcs([ft[0] for ft in fts])
         ar_out,footprint = reproject_and_coadd([ft[0] for ft in fts],
             wcs_out, shape_out=shape_out, reproject_function=reproject_interp,
             match_background=True)
         h_out = fts[0][0].header.copy()
         for key in list(h_out.keys()):
            if key in ['CD1_1','CD2_2','CD1_2','CD2_1'] or key[:2] == 'PV':
               h_out.remove(key)
         h_out.update(wcs_out.to_header())
         newhdu = fits.PrimaryHDU(ar_out, header=h_out)
         ret.append((fits.HDUList([newhdu]),fts[idx])) 
      else:
         fts = fits.open(urls[0])
         ret.append((fts,fts))

   return ret


def getStarCat(ra, dec, radius):
   '''Get a list of SM stars plus their photometry.'''

   templ = "https://skymapper.anu.edu.au/sm-cone/public/query?RA={}&DEC={}"\
          "&SR={}&VERB=3&RESPONSEFORMAT=CSV"
   url = templ.format(ra, dec, radius)

   tab = ascii.read(url)
   if len(tab) == 0:
      return None
   
   tab = tab[tab['class_star'] > 0.8]
   tab = tab['object_id', 'raj2000','dej2000','g_psf','e_g_psf','r_psf',
         'e_r_psf', 'i_psf','e_i_psf']
   tab.rename_column('object_id','objID')
   tab.rename_column('raj2000', 'RA')
   tab.rename_column('dej2000', 'DEC')
   tab.rename_column('g_psf','gmag')
   tab.rename_column('e_g_psf','gerr')
   tab.rename_column('r_psf','rmag')
   tab.rename_column('e_r_psf','rerr')
   tab.rename_column('i_psf','imag')
   tab.rename_column('e_i_psf','ierr')

   gids = ~np.isnan(tab['rmag'])
   for tag in ['rmag','imag','gmag','rerr','ierr','gerr']:
      gids = gids*getattr(tab[tag], 'mask', 1)

   gids = np.greater(tab['gmag'], 0)*np.less(tab['gmag'],20)
   gids = gids*np.greater(tab['rmag'], 0)*np.less(tab['rmag'],20)
   gids = gids*np.greater(tab['imag'], 0)*np.less(tab['imag'],20)

   tab = tab[gids]
   return tab


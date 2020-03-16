'''Module for downloading POSS data (image cutout).'''

import numpy as np
from astropy.io import fits

def getFITS(ra, dec, size):
   '''Retrieve the FITS files from POSS server, centered on ra,dec
   and with given size.

   Args:
      ra (float):  RA in degrees
      dec (float):  DEC in degrees
      size (float):  size of FOV in degrees

   Returns:
      list of FITS instances
   '''
   templ = 'http://archive.stsci.edu/cgi-bin/dss_search?r={}&d={}&h={}&w={}'
   size = size*60   # arc-min
   # max size form image server... need to work on more mosaicking later.
   if size > 60: size = 60

   url = templ.format(ra,dec,size,size)
   fts = fits.open(url)
   return fts


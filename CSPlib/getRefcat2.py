'''Adapted from code by Priscilla Pessi. This modules is for querying the 
RefCat2 catalog for magnitudes of field stars.'''

from .config import getconfig
import os
import glob
from . import refcat
from astropy import table
import numpy as np

cfg = getconfig()

# Try to figure out where REfcat is
if getattr(cfg.data, 'refcatdir', None) is not None:
   refcatdir = cfg.data.refcatdir
elif 'REFCATDIR' in os.environ:
   refcatdir = os.environ['REFCATDIR']
else:
   refcatdir = '/Volumes/ExtData1/RefCat2'


def getStarCat(ra, dec, radius, mmin=-10, mmax=100):
   '''Get a list of refcat2 stars plus their photometry.

   Args:
      ra (float); RA in decimal degrees
      dec (float):  DEC in decima degress
      radius (float):  cone radius in decimal degrees
      mmin (float): minimum magnitude to return
      mmax (float): maximum magnitude to return

   Returns:
      astropy.table with catalog data
   '''

   if not mmin < mmax:
      raise ValueError("mmin must be less than mmax")

   tabs = []
   if mmin < 16:
      tabs.append(refcat.RefcatQuery(ra, dec, 1, radius, radius, mmax, 0,
         [os.path.join(refcatdir, '00_m_16')], 'rc2'))
   if mmin < 17 and mmax > 16:
      tabs.append(refcat.RefcatQuery(ra, dec, 1, radius, radius, mmax, 0,
         [os.path.join(refcatdir, '16_m_17')], 'rc2'))
   if mmin < 178 and mmax > 17:
      tabs.append(refcat.RefcatQuery(ra, dec, 1, radius, radius, mmax, 0,
         [os.path.join(refcatdir, '17_m_18')], 'rc2'))
   if mmax > 18:
      tabs.append(refcat.RefcatQuery(ra, dec, 1, radius, radius, mmax, 0,
         [os.path.join(refcatdir, '18_m_19')], 'rc2'))
   tab = table.vstack(tabs)
   gids = np.greater_equal(tab['r'], mmin)*np.less_equal(tab['r'], mmax)
   tab = tab[gids]

   tab['objID'] = np.arange(1, len(tab)+1)
   tab = tab['objID','RA','Dec','g','dg','r','dr','i','di']
   tab.rename_column('Dec','DEC')
   for filt in ['g','r','i']:
      tab.rename_column(filt, filt+'mag')
      tab.rename_column('d'+filt, filt+'err')
   return tab


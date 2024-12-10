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

use_local = False
if os.path.isdir(refcatdir):
   if os.path.isdir(os.path.join(refcatdir, '16_m_17')):
      # Okay, looks like we got it
      use_local = True
   
if not use_local:
   # refcatdir not found. Can use do remote queries?
   MASTuser = getattr(cfg.remote, 'MASTuser', None)
   MASTpasswd = getattr(cfg.remote, 'MASTpasswd', None)
   if MASTuser is not None and MASTpasswd is not None:
      try:
         from mastcasjobs import MastCasJobs
         jobs = MastCasJobs(username=MASTuser, password=MASTpasswd, 
                            context="HLSP_ATLAS_REFCAT2")
      except:
         jobs = None
   else:
      jobs = None


def getStarCat(ra, dec, radius, mmin=-10, mmax=100):
   '''Get a list of refcat2 stars plus their photometry.

   Args:
      ra (float); RA in decimal degrees
      dec (float):  DEC in decima degress
      radius (float):  cone radius in decimal degrees
      mmin (float): minimum magnitude to return
      mmax (float): maximum magnitude to return

   Returns:
      astropy.table with catalog data or None if something went wrong
   '''


   if not mmin < mmax:
      raise ValueError("mmin must be less than mmax")
   ra = float(ra)*1
   dec = float(dec)*1

   if use_local:
      tabs = []
      if mmin < 16:
         #print(ra,dec)
         tabs.append(refcat.RefcatQuery(ra, dec, 1, radius, radius, mmax, 0,
            [os.path.join(refcatdir, '00_m_16')], 'rc2', verbose=0))
      if mmin < 17 and mmax > 16:
         tabs.append(refcat.RefcatQuery(ra, dec, 1, radius, radius, mmax, 0,
            [os.path.join(refcatdir, '16_m_17')], 'rc2', verbose=0))
      if mmin < 178 and mmax > 17:
         tabs.append(refcat.RefcatQuery(ra, dec, 1, radius, radius, mmax, 0,
            [os.path.join(refcatdir, '17_m_18')], 'rc2', verbose=0))
      if mmax > 18:
         tabs.append(refcat.RefcatQuery(ra, dec, 1, radius, radius, mmax, 0,
            [os.path.join(refcatdir, '18_m_19')], 'rc2', verbose=0))
      tab = table.vstack(tabs)
      gids = np.greater_equal(tab['r'], mmin)*np.less_equal(tab['r'], mmax)
      tab = tab[gids]
 
      tab['objID'] = np.arange(1, len(tab)+1)
      tab = tab['objID','RA','Dec','g','dg','gcontrib','r','dr','rcontrib',\
                'i','di','icontrib']
      tab.rename_column('Dec','DEC')
      for filt in ['g','r','i']:
         tab.rename_column(filt, filt+'mag')
         tab.rename_column('d'+filt, filt+'err')
         tab.rename_column(filt+'contrib', filt+'con')
      return tab
   elif jobs is not None:
      query = "select r.objid,r.RA,r.Dec,r.g,r.dg,r.gcontrib,r.r,r.dr,"\
              "r.rcontrib,r.i,r.di,r.icontrib "\
              "from refcat2 as r, dbo.fGetNearbyObjEq({},{},{}) as n "\
              "where r.objid=n.objid".format(ra,dec,radius)
      tab = jobs.quick(query, task_name='my task')
      gids = np.greater_equal(tab['r'], mmin)*np.less_equal(tab['r'], mmax)
      tab = tab[gids]
      
      tab['objID'] = np.arange(1, len(tab)+1)
      tab = tab['objID','RA','Dec','g','dg','gcontrib','r','dr','rcontrib',\
                'i','di','icontrib']
      tab.rename_column('Dec','DEC')
      for filt in ['g','r','i']:
         tab.rename_column(filt, filt+'mag')
         tab.rename_column('d'+filt, filt+'err')
         tab.rename_column(filt+'contrib', filt+'con')
      #gids = np.greater_equal(tab['r'], mmin)*np.less_equal(tab['r'], mmax)
      return tab

   else:
      print("Error:  No local refcat catalog found and remote access failed")
      return None
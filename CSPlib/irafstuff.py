'''module for IRAF replacements.'''
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.io import fits
from numpy import *
from .npextras import mode
import os
from glob import glob
import re

lco = EarthLocation.of_site('lco')

secpat = re.compile(r"\[([0-9]+|\*):([0-9]+|\*),([0-9]+|\*):([0-9]+|\*)\]")



def sec2slice(sec):
   '''For an IRAF-like section, parse it and return a slice. Note that
   IRAF slices are transposed from numpy slices. This function does that
   transpose for you.'''
   res = secpat.findall(sec)
   if len(res) == 0:
      return None
   if len(res[0]) != 4:
      return None
   args = []
   for i in [2,3,0,1]:     # transpose
      if res[0][i] == '*':
         args.append(None)
      else:
         args.append(int(res[0][i]))
   return (slice(args[0],args[1]),slice(args[2],args[3]))

def getInputList(l):
   '''Get an input list. Following this logic. l can be:
   str:  treat as a glob pattern
   str prefixed with '@': get list from file
   list of str:  list of filenames
   list of fits:  as is
   
   Args:
      l (multi):  input list. See text above
      
   Returns:
      fitslist:  list of FITS files.
   '''

   if isinstance(l,str):
      if l[0] == '@':
         if not os.path.isfile(l[1:]):
            raise ValueError('File {} not found'.format(l[1:]))
         with open(l[1:], 'r') as fin:
            flist = fin.readlines()
            flist = [f.strip() for f in flist]
      else:
         flist = glob(l)
         if len(flist) == 0:
            raise FileNotFoundError("No files found matching {}".format(l))
   elif isinstance(l, list):
      if isinstance(l[0], str):
         flist = l
      elif isinstance(l[0], fits.HDUList):
         return l
      else:
         raise TypeError('Invalid input type')
   else:
      raise TypeError('Invalid input type')
   ftslist = [fits.open(f) for f in flist]
   return ftslist


def wairmass_for_lco_images(ra, dec, equinox, dateobs, utstart, exptime,
      scale=750.):
   '''Compute the effective airmass for observations at LCO. 

   Args:
      ra(str):       Right-ascention in the format hh:mm:ss.s
      dec(str):      Declination in the format +dd:mm:ss.s
      equinox(float): Equinox in decimal years
      dateobs(str): Date of observation in the format yyyy-mm-dd
      utstart(str):    Start UT time in the format hh:mm:ss
      exptime(float): Exposure time in seconds
      scale (float): Scale hight of atmosphere

   Returns:
      (AM, ST, UTmid):  Airmass (float), 
                        sidereal time (astropy.Time)
                        UT time at mid-exposure (astropy.Time)'''

   eq = Time(equinox, format='decimalyear')
   obj = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), equinox=eq)
   obstime = Time("{}T{}".format(dateobs, utstart), format='isot', scale='utc',
         location=lco)
   utmid = obstime + exptime*u.s/2
   tsid = utmid.sidereal_time('apparent')
   utend = obstime + exptime*u.s

   altaz = obj.transform_to(AltAz(obstime=[obstime, utmid, utend],
      location=lco))
   elev = altaz.alt.value*pi/180
   x = scale*sin(elev)
   AMs = sqrt(x**2 + 2*scale + 1) - x
   AM = (AMs[0] + 4*AMs[1] + AMs[2])/6
   return(AM, tsid, utmid)

def computeCenter(cube, axis=0, mclip=False, exclude_ends=False, mask=None):
   '''Compute the center of an array. 

   Args:
      cube (array):  array to compute the center.
      axis (int):  which axis to compute on
      mclip (bool):  If true, compute median, otherwise mean
      exclude_ends (bool):  If true, omit the ends of the array along axis
      mask (array):  array of True/False for which values to keep

   Returns:
      center:  an array with one less dimension.
   '''

   if mclip:
      if mask is not None:
         return ma.median(ma.masked_array(cube, ~mask))
      return median(cube, axis=axis)
   if exclude_ends:
      sortcube = sort(cube, axis=0)
      return(mean(sortcube[1:-1], axis=0))
   return sum(cube*mask, axis=axis)/sum(mask, axis=axis)

def imcombine(inp, combine='average', reject='avsigclip', statsec=None,
      gain=1, rdnoise=0, lsigma=3, hsigma=3, pclip=-0.5, nlow=0, nhigh=1,
      nkeep=1, mclip=False, weight=None, scale=None, 
      expname='EXPTIME', verbose=False):
   '''Combine images, a la IRAF imcombine.

   Args:
      inp (list or str): Input list of images (prefix @ means get from file)
      compbine (str):  'average' or 'median'
      reject (str): 'avsigclip', 'sigclip', 'minmax', or 'none'
      statsec (str): section for statistics in the form [N:M,n:m]
      gain (float): conversion from data units to electrons
      rdnoise (float): read noise in electrons
      lsigma (float): lower sigma for clipping
      hsigma (float): upper sigma for clipping
      pclip (float):  percentile clipping (see IRAF docs for useage)
      nlow (int): number of low pixels to reject for minmax reject
      nhigh (int): number of high pixels to reject for minmax reject
      nkeep (int): minimum number of pixels to keep after rejection
      mcplip (bool): use average (false)  or median (true) for sigma clipping
      weight (array): weight array to use in weighted average
      scale (str): scale the images ('mode','median','mean','exposure')
      expname (str): exposure keyader keyword

   Returns:
      FITS HDUList of combined image, with header copied from first input
      image, NCOMBINE updated, and comments added.'''

   ftslist = getInputList(inp)
   Nim = len(ftslist)
   if verbose: print("Working on {} images".format(Nim))
   try:
      cube = asarray([f[0].data for f in ftslist])
   except:
      raise ValueError("Cannot create a data cube. Not all consistent shape?")

   if statsec is not None:
      if isinstance(statsec,str) or isinstance(statsec,bytes):
         slc = sec2slice(statsec)
         if slc is None:
            raise ValueError("Cannot parse the statsec {}".format(statsec))
      else:
         slc = (slice(statsec[2],statsec[3]),slice(statsec[0],statsec[1]))
   else:
      slc = (slice(None,None),slice(None,None))

   # Figure out the scales
   if scale is None:
      scales = ones(cube.shape[0])
   elif scale == 'mode':
      #from scipy.stats import mode
      scales = array([1.0/mode(cube[i][slc]) \
            for i in range(cube.shape[0])])
   elif scale == 'median':
      scales = array([1.0/median(cube[i][slc],axis=None) \
            for i in range(cube.shape[0])])
   elif scale == 'mean':
      scales = array([1.0/mean(cube[i][slc],axis=None) \
            for i in range(cube.shape[0])])
   elif scale == 'exposure':
      scales = array([1.0/f[0].header[expname] for f in flist])
   else:
      raise ValueError('Unrecognized scale method {}'.format(scale))

   cube = cube * scales[:,newaxis,newaxis]

   # Make a mask for rejections
   if reject == 'minmax':
      if nlow + nhigh >= Nim:
         raise ValueError("Error:  nlow +nhigh > number of images")
      sids = argsort(cube, axis=0)
      mask = greater(sids, nlow-1) * less(sids, Nim-nhigh)
      if verbose: print("Using min/max rejected {} pixels".format(sum(~mask)))
   elif reject == 'sigclip':
      if Nim < 3:
         raise ValueError("Error:  sigclip needs at least 3 images")
      # Compute average or median
      center = computeCenter(cube, axis=0, mclip=mclip, exclude_ends=True,
            mask=None)
      # sigma about center:
      resids = cube - center[newaxis,:,:]
      sigma = std(resids, axis=0)
      mask = greater(resids, -sigma*lsigma)*less(resids,sigma*hsigma)
      keep = sum(mask, axis=0)
      if verbose: print("Using sigclip rejected {} pixels".format(sum(~mask)))
      while sometrue(greater(keep, 3)):
         center = computeCenter(cube, axis=0, mclip=mclip, exclude_ends=False,
               mask=mask)
         resids = cube-center[newaxis,:,:]
         sigma = std(resids, axis=0)
         mask = greater(resids, -sigma*lsigma)*less(resids, sigma*hsigma)
         if alltrue(equal(sum(mask, axis=0) - keep, 0)):
            # No change, we're done
            break
         keep = sum(mask, axis=0)
         if verbose: print("   iterate rejected {} pixels".format(sum(~mask)))

   elif reject == 'avsigclip':
      # See IRAF docs
      # first, we compute the average gain across rows
      center = computeCenter(cube, axis=0, mclip=mclip, exclude_ends=True,
            mask=None)
      v = var(cube-center[newaxis,:,:], axis=0)
      gain = mean(v/center, axis=1)
      sigma = sqrt(gain[:,newaxis]*center)
      if verbose: print("AVSIGCLIP:  gain:{}, sigma:{}-{}".format(mean(gain),
         sigma.min(), sigma.max()))

      # Now have sigma for each pixel
      resids = cube - center[newaxis,:,:]
      mask = greater(resids, -sigma*lsigma)*less(resids,sigma*hsigma)
      if verbose: print("Using avsigclip rejected {} pixels".format(sum(~mask)))
      keep = sum(mask, axis=0)
      while sometrue(greater(keep, 3)):
         center = computeCenter(cube, axis=0, mclip=mclip, exclude_ends=False,
               mask=mask)
         resids = cube-center[newaxis,:,:]
         mask = greater(resids, -sigma*lsigma)*less(resids, sigma*hsigma)
         if alltrue(equal(sum(mask, axis=0) - keep, 0)):
            # No change, we're done
            break
         keep = sum(mask, axis=0)
         if verbose: print("   iterate rejected {} pixels".format(sum(~mask)))
   else:
      raise NotImplemented("rejection method {} not implemented yet".format(
         reject))
   if reject in ['sigclip','avsigclip']:
      keep = sum(mask, axis=0)
      if sometrue(keep < nkeep):
         resids = absolute(resids)
         sids = argsort(resids, axis=0)
         # number that need to be added
         numadd = where(keep < nkeep, nkeep-keep, 0)
         ii,jj = nonzero(numadd > 0)
         if verbose: 
            print("Need to work on putting back data for {} pixels".format(
               len(ii)))
         for i,j in zip(ii,jj):
            sids = argsort(resids[:,i,j])
            resid = resids[:,i,j][sids[numadd[i,j]]]
            gids = less_equal(resids[:,i,j], resid)
            mask[gids,i,j] = True
      keep = sum(mask, axis=0)

   # Now do the math:
   if combine == 'average':
      if weight is None:
         weight = mask*1.0
      else:
         weight = weight*mask
      retdata = sum(weight*cube, axis=0)/sum(weight, axis=0)
   elif combine == 'median':
      retdata = ma.median(ma.masked_array(cube, mask=~mask), axis=0)
      retdata = ma.filled(retdata, fill_value=1.0)
   else:
      raise NotImplemented("Combine method {} not implemented yet".format(
         combine))

   phdu = fits.PrimaryHDU(retdata, header=ftslist[0][0].header)
   Ncombines = 0
   for ft in ftslist:
      if 'NCOMBINE' in ft[0].header:
         Ncombines += ft[0].header['NCOMBINE']
      else:
         Ncombines += 1
   phdu.header['NCOMBINE'] = Ncombines
   phdu.header['COMMENT'] = "Imcombine using reject={} and combine={}".format(
         reject, combine)
   return fits.HDUList([phdu])

#def imstatistics(images, fields=None, lower=None, upper=None, nclip=0, 
#      lsigma=3.0, usigma=3.0, binwidth=0.1):
#   '''Do image statistics and return a table of results.'''
#   ftslist = getInputList(images)
#   Nim = len(ftslist)
#
#   funcs = {'
#
#   if fields is None:
#      fields = ['image','npix','mean','midpt','mode','stddev','skew','kurtosis',
#                'min','max']
#   arr = []
#   for fts in ftslist:
#      for field in fields:
#

'''A module for performing photometry. Uses photutils which, at this point,
seems to only do aperture photometry reliably.'''

from .config import config
from .tel_specs import getTelIns

from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.stats import gaussian_fwhm_to_sigma as FtoS
import astropy.units as u
from astropy.io import ascii,fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.visualization import simple_norm

from photutils.segmentation import make_source_mask
from photutils import SkyCircularAperture, SkyCircularAnnulus
from photutils import aperture_photometry
from photutils.centroids import centroid_com,centroid_1dg
from photutils.psf import BasicPSFPhotometry, IntegratedGaussianPRF, DAOGroup
#from photutils.psf import EPSFBuilder, extract_stars
from .myepsfbuilder import FlowsEPSFBuilder as EPSFBuilder
from photutils.psf import extract_stars
from photutils.background import MMMBackground, MedianBackground,Background2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import models
from scipy.optimize import least_squares

from matplotlib import pyplot as plt
from astropy.visualization import simple_norm

from warnings import catch_warnings, simplefilter
import os
from numpy import *
from .npextras import between
from astropy.table import vstack,Table
import tempfile
from copy import deepcopy

try:
   import pymc3 as pymc
except:
   pymc = None

try:
   import astroscrappy
except:
   astroscrappy = None

def recenter(xs, ys, data, cutoutsize=40, method='1dg'):
   '''Given initial guesses for star positions, re-center.

   Input:
      xs,ys (arrays):  the inital x,y pixel guesses
      data (2d array):  The data with centroids
      cutoutsize(int):  the boxsize of the data to cut out and centroid
      method (str):  which centroiding method:  'com','1dg','2dg'
   Returns:
      x,y,flag (arrays):   x,y: output centroid coordinates
                          flag:  array of flags
                                 (0-OK, 1-Off frame, 2-no convergence)
   '''
   flags = []; xout = []; yout = []
   if cutoutsize % 2 == 0: cutoutsize += 1   # make sure odd
   interv = cutoutsize//2                    # integer division!!!

   for i in range(len(xs)):
      if not (interv < xs[i] < data.shape[1]-interv and\
            interv < ys[i] < data.shape[0]-interv):
         xout.append(xs[i])
         yout.append(ys[i])
         flags.append(1)    # outside data array
      else:
         x = int(xs[i])
         y = int(ys[i])
         sdata = data[y-interv:y+interv+1, x-interv:x+interv+1]
         mn,md,st = sigma_clipped_stats(sdata, sigma=2.)
         if method == 'com':
            xx,yy = centroid_com(sdata - md)
         elif method == '1dg':
            xx,yy = centroid_1dg(sdata - md)
         elif method == '2dg':
            xx,yy = centroid_1dg(sdata - md)
         elif method == 'quad':
            xx,yy = centroid_quadratic(sdata - md)
         if not (0 < xx < cutoutsize and 0 < yy < cutoutsize):
            xout.append(xs[i])
            yout.append(ys[i])
            #outtab[i]['x'] = tab[i]['x']
            #outtab[i]['y'] = tab[i]['y']
            flags.append(2)
         else:
            #outtab[i]['x'] = x - interv + xx
            #outtab[i]['y'] = y - interv + yy
            xout.append(x - interv + xx)
            yout.append(y - interv + yy)
            flags.append(0)
   return array(xout),array(yout),array(flags)

def objfunc(p, mod, x, y, z, w):
   res = (z - mod.evaluate(x, y, p[0], p[1], p[2], p[3], p[3], 0))*w
   return res.ravel()

def centroid2D(data, i0, j0, fwhm0, radius, var=None, gain=1, rdnoise=0,
                     bg=0, mask=None, axis=None, profile='Gauss'):
   '''Using a 1D Gausssian, fit the centroid and FWHM of a stellar object.
   
   Args:
      data (ndarray):  the FITS data (full frame) to fit
      i0,j0 (float):   initial position of pixel centroid
      fwhm0 (float):   initial value for FWHM
      radius (float):  the size of cutout to fit.
      var (ndarray):   variance array for data. If None, compute it
      gain (float):    gain of data (used for errors if var is None)
      rdnoise (float): read noise in e- (used for errors if var is None)
      bg (float):      background value (used for errors if var is None
                       and data was background-subtracted).
      axis (mpl.Axes): MPL axis istance. If not None, plot radial profile
      profile(str):    which profile to use ('Gauss' or 'Moffat')

   Returns:
      success,xfit,yfit,fwhm,rchisq
         success (bool):     True if successful fit
         xfit,yfit (float):  fit pixel positions
         fhwm (float):       fit FWHM in pixels
         rchisq (float:      reduced-chi-square of fit
         peakSNR (flaot):    peak S/N of subraster
   '''
   jm,im = data.shape
   if not radius < i0 < im-radius or not radius < j0 < jm-radius:
      return False,i0,j0,fwhm0,-1,-1
   imin = max(0, int(i0)-radius);  imax = min(int(i0)+radius, im)
   jmin = max(0, int(j0)-radius);  jmax = min(int(j0)+radius, jm)

   subdat = deepcopy(data[jmin:jmax, imin:imax])
   if mask is not None:
      submask = deepcopy(mask[jmin:jmax, imin:imax])

   if var is None:
      # Make noise from subdat
      subvar = (subdat+bg)/gain + rdnoise**2/gain**2
   else:
      subvar = deepcopy(var[jmin:jmax, imin:imax])
   
   bids = ~(isfinite(subdat) & greater(subvar, 0))
   subdat[bids] = 0
   subvar[bids] = 1
   norm = subdat[~bids].max()
   subdat /= norm
   subvar /= norm**2

   peakSNR = power(subvar,-0.5).max()
   weights = power(subvar, -0.5)*~bids    # zero-weight the bad data

   # Setup fitting
   fitter = LevMarLSQFitter(calc_uncertainties=True)
   if profile.lower() == 'gauss':
      g2 = models.Gaussian2D(amplitude=1.0, x_mean=radius, y_mean=radius, 
                             x_stddev=fwhm0*FtoS)
      #bounds = (array([0.1, radius/2, radius/2, fwhm0/10*FtoS]),
      #          array([2.0, radius*2, radius*2, fwhm0*10*FtoS]))
      g2.amplitude.bounds = (0.1, 2.0)
      g2.x_mean.bounds = (radius/2, radius*2)
      g2.y_mean.bounds = (radius/2, radius*2)
      g2.x_stddev.bounds = (fwhm0/10*FtoS, 5*fwhm0*FtoS)
      g2.y_stddev.tied = lambda model: model.x_stddev
      g2.theta.fixed = True
   else:
      g2 = models.Moffat2D(amplitude=1.0, x_0=radius, y_0=radius, 
                             gamma=fwhm0, alpha=1.5)
      #bounds = (array([0.1, radius/2, radius/2, fwhm0/10*FtoS]),
      #          array([2.0, radius*2, radius*2, fwhm0*10*FtoS]))
      g2.amplitude.bounds = (0.1, 2.0)
      g2.x_0.bounds = (radius/2, radius*2)
      g2.y_0.bounds = (radius/2, radius*2)
      g2.gamma.bounds = (fwhm0/10, 5*fwhm0)

   yy,xx = mgrid[:subdat.shape[1], :subdat.shape[0]]
   fit = fitter(g2, x=xx, y=yy, z=subdat, weights=weights)
   #p0 = [1.0, radius, radius, fwhm0*FtoS]
   #res = least_squares(objfunc, p0, args=(g2, xx, yy, subdat, weights))
   #model = g2.evaluate(xx, yy, res.x[0], res.x[1], res.x[2], res.x[3], res.x[3], 0)

   model = fit(xx,yy)
   chisq = sum((subdat-model)**2/subvar)
   rchisq = chisq / (subdat.shape[0]*subdat.shape[1] - 4)

   if getattr(g2, 'x_mean', None) is not None:
      xm = fit.x_mean
      ym = fit.y_mean
      fwhm = fit.x_stddev/FtoS
   else:
      xm = fit.x_0
      ym = fit.y_0
      fwhm = fit.fwhm

   # Now check to see if any masked pixels are "close" to the core
   if mask is not None:
      rads = sqrt((xx - xm)**2 + (yy - ym)**2)
      gids = less(rads, 2*fwhm*FtoS).ravel()
      if sometrue(submask.ravel()[gids]):
         return(False, imin+xm, jmin+ym, fwhm, rchisq, peakSNR)

   if axis is not None:
      #rads = sqrt((xx - res.x[1])**2 + (yy - res.x[2])**2)
      rads = sqrt((xx - xm)**2 + (yy - ym)**2)
      axis.plot(rads.ravel(), subdat.ravel(), '.', color='k', alpha=0.5)
      sids = argsort(rads.ravel())
      axis.plot(rads.ravel()[sids], model.ravel()[sids], '-', color='red', 
                alpha=0.5, zorder=1000)

   return (True,imin+xm, jmin+ym, fwhm, rchisq, peakSNR)

class BasePhot:

   def __init__(self, ftsfile, tel='SWO', ins='NC', sigma=None, mask=None):
      '''Initialize this photometry class with a tel/ins configuration
      and FITS file.
      
      Args: 
         ftsfile (str or FITS): The FITS file to deal with
         tel (str):  telescope code (e.g., SWO)
         ins (str):  instrument code (e.g., NC)
         sigma (str or FITS): The FITS file with error (noise) map
         mask (str or FITS): The optional FITS file with mask (True=Bad) 
      
      Returns:
         PSFPhot instance.
      '''
      self.cfg = getTelIns(tel,ins)
      if not os.path.isfile(ftsfile):
         raise ValueError("Error:  not such file {}".format(ftsfile))
      if isinstance(ftsfile, str):
         self.ftsobj = fits.open(ftsfile)
         self.ftsfile = ftsfile
      else:
         self.ftsobj = ftsfile
         self.ftsfile = None

      self.head = self.ftsobj[0].header
      self.data = self.ftsobj[0].data
      with catch_warnings():
         simplefilter('ignore')
         self.wcs = WCS(self.head)

      self.RAs = None
      self.DEs = None
 
      self.parse_header()
      if sigma is None:
         if 'ERR' in self.ftsobj:
            self.error = self.ftsobj['ERR'].data
         else:
            self._makeErrorMap()
      else:
         if isinstance(sigma, str):
            sigma = fits.open(sigma)
         self.error = sigma[0].data

      if mask is None:
         if 'BPM' in self.ftsobj:
            self.mask = self.ftsobj['BPM'].data
         else:
            self.mask = self._makeMask()
      else:
         if isinstance(mask, str):
            mask = fits.open(mask)
         self.mask = mask[0].data

      self.centroids = None
      self.phot = None

      self.background = None

   def _parse_key(self, key, fallback=None):
      '''Given a key, we try to figure out what value it sould have. First,
      we check if the key exists in the config dict. If it is a string has
      starts with '@', we take it from the fits header. Otherwise, it
      is taken as the value from the config dict. If the key is not in dict
      we can fallback if it is not None. Othewise, raise an exception.'''
      val = self.cfg.get(key, None)
      if val is not None:
         if isinstance(val,str) and val.find('@') == 0:
            fitskey = val[1:]
            if fitskey not in self.head:
               if fallback is not None:
                  print("Warning: Header keyword {}, not found, using fallback={}".format(
                     fitskey, fallback))
                  return fallback
               raise KeyError("header keyword {} not found".format(fitskey))
            return self.head[fitskey.upper()]
         else:
            return val
      else:
         if fallback is not None:
            print("Warning: couldn't figure out value for {}, using fallback={}".format(
               key, fallback))
            return fallback
         else:
            raise KeyError("Option {} not found in data section".format(key))

   def parse_header(self):
      '''Parse the FITS header for information we'll need, falling back on
      defaults or values from the config file.'''
      self.exposure = self._parse_key("exposure", fallback=1)
      self.filter = self._parse_key("filter", fallback='X')
      self.date = self._parse_key("date", fallback=0.0)
      self.gain = self._parse_key("gain", fallback=1.0)
      self.ncombine = self._parse_key("ncombine", fallback=1)
      self.rdnoise = self._parse_key("rnoise", fallback=0.0)
      self.airmass = self._parse_key("airmass", fallback=1.0)
      self.object = self._parse_key("object", fallback='SNobj')
      self.scale = self._parse_key("scale", fallback=1.0)
      self.datamax = self._parse_key("datamax", fallback=65000.)
      self.meansky = self._parse_key("meansky", fallback=0.)

   def _makeErrorMap(self):
      '''Compute the error in the data based on gain, readnoise, etc.'''
      # effective gain:
      egain = self.gain*self.ncombine
      with catch_warnings():
         simplefilter("ignore")
         self.error = where(self.data > 0,
               sqrt((self.data+self.meansky)/egain + self.rdnoise**2/self.gain**2),
               self.rdnoise/self.gain)

   def _makeMask(self):
      # Look for bad pixels.  Start with NaN's
      mask = isnan(self.data)

      # Next if datamax is specified:
      mask = mask | greater(self.data, self.datamax)

      return mask

   def CRReject(self, fix=False, sigclip=4.5, cleantype='meanmask'):
      '''Use LA Cosmic (implemented by the astroscrappy module) to flag
      and optionally fix cosmic rays of single images.
      
      Args:
         fix (boolean):  If True, replace the cosmic rays 
         cleantype (str): Which type of cleaning algorithm. Choices are:
                          "median", "medmask", "meanmask", or "idw"
                          (see astroscrappy module for details)
         sigclip (float): Laplacian-to-noise limit for CR rejection

      Returns:
         None

      Effects:
         Updates self.mask to mask out cosmic rays. If fix=True, then
         update self.data to cleaned version.'''

      if astroscrappy is None:
         raise ModuleNotFoundError("Error:  you need astroscrappy to do"\
               " CR rejection")
      m,a = astroscrappy.detect_cosmics(self.data, sigclip=sigclip,
                                        invar=power(self.error,2),
                                        cleantype=cleantype)
      self.mask = self.mask | m
      if fix:
         self.data = a

      return

   def model2DBackground(self, boxsize=50, nsigma=2, npixels=10):
      '''Construct a 2D background estimate of the image.

      Args:
         boxsize (int):  Box size used to make BG esimate. The larger, the
                         better the stats, but more coarse the estimate.

      Returns:
         None

      Effects:
         self.background set to resulting 2D background instance.
      '''
      mask = make_source_mask(self.data, nsigma=nsigma, npixels=npixels, 
                              dilate_size=11, mask=self.mask)
      sigma_clip = SigmaClip(sigma=3.)
      bkg_estimator = MedianBackground()
      bkg = Background2D(self.data, (boxsize,boxsize), filter_size=(3,3),
                         sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                         mask=mask)
      self.background = bkg

   def loadObjCatalog(self, table=None, filename=None, racol='col2',
         deccol='col3', objcol='col1'):
      '''Given a catalog filename, load it and store info.
      Args:
         table (astropy.table):  source catalog as a table
         filename (str):  source catalog as a file
         racol (str):  column name for RA
         deccol (str): column name for DEC
         objcol (str); column name for object number
         
      Returns:
         None

      Effects:
         self.objs, self.RAs, self.DEs populated with data
      '''
      if filename is None and table is None:
         if self.ftsfile is not None:
            if os.path.isfile(self.ftsfile.replace('.fits','.cat')):
               filename = self.ftsfile.replace('.fits','.cat')
            else:
               raise ValueError("you must specify either a file or table")
         else:
            raise ValueError("you must specify either a file or table")

      if table is not None:
         cat = table
      else:
         cat = ascii.read(filename)

      # Check for needed info
      if objcol not in cat.colnames:
         raise ValueError(
               "Error: object column {} not found in data file".format(objcol))
      self.objs = cat[objcol]
      
      if racol not in cat.colnames:
         raise ValueError(
               "Error: RA column {} not found in data file".format(racol))
      self.RAs = cat[racol]

      if deccol not in cat.colnames:
         raise ValueError(
               "Error: DEC column {} not found in data file".format(deccol))
      self.DEs = cat[deccol]

   def fitFWHM(self, SNRmin= 5, Nmin=5, plotfile=None, profile='Gauss'):
      '''Fit simple 2D symmetric Gaussian profiles to the LS stars and get the
      average FWHM of the image in arc-seconds.

      Args:
         SNRmin (float):       Minimum peak S/R radio for using to compute FWHM
         Nmin (int):           Minimum number of stars needed to compute FWHM
         plotfile(str):        Output radial plot of the profile and fit 
         profile(str):         Which profile to use 'Gauss' or 'Moffat'

      Returns:
         fwhm, table
             fwhm:  the full-width at half-maximum in arc-sec
             tab:   table of positions (obj, RA, DEC, xpix, ypix, fwhm, rchisq)

      '''
      if self.RAs is None:
         raise ValueError("You need to load a catalog first")

      if self.background is None:
         bg = median(self.data.ravel())
      else:
         bg = self.background.background

      if plotfile is not None:
         fig,ax = plt.subplots()
         ax.set_xlabel('radial distance (pixels)')
         ax.set_ylabel('normalized counts')
      else:
         ax = None

      # list of stars to work with
      xs,ys = self.wcs.wcs_world2pix(self.RAs, self.DEs, 0)
      gids = []; xout = []; yout = []; fwhms = []; rchisqs = []; snrs=[]
      for i in range(len(xs)):
         if self.objs[i] < 1:
            # SN, so don't use that
            gids.append(False)
            continue
         stat,x,y,fwhm,rchisq,snr = centroid2D(self.data - bg, 
                                  xs[i], ys[i], 
                                  1.0/self.scale, int(10/self.scale), 
                                  var=self.error**2, axis=ax, profile=profile)
         if not stat:
            gids.append(False)
            continue
         gids.append(True)
         xout.append(x)
         yout.append(y)
         fwhms.append(fwhm*self.scale)   # In arc-sec
         rchisqs.append(rchisq)
         snrs.append(snr)

      tab = Table([self.objs[gids],self.RAs[gids],self.DEs[gids],xout,yout,
                 fwhms,rchisqs, snrs], 
                 names=['objID','RA','DEC','xfit','yfit','fwhm','rchisq','snr'])
      tab['xfit'].info.format = "%.3f"
      tab['yfit'].info.format = "%.3f"
      tab['fwhm'].info.format = "%.2f"
      tab['rchisq'].info.format = "%.3f"
      tab['snr'].info.format = "%.2f"

      #if len(tab) < Nmin:
      #   raise ValueError("Less than Nmin ({}) stars fit".format(Nmin))

      mask = (tab['snr'] > SNRmin)
      if sum(mask) > Nmin:
         fwhm = median(tab['fwhm'][mask])
      elif len(tab) > Nmin:
         sids = argsort(tab['snr'])
         fwhm = median(tab['fwhm'][sids][:Nmin])
         for i in range(Nmin):
            mask[sids[i]] = True
      elif len(tab) > 0:
         idx = argmax(tab['snr'])
         fwhm = tab['fwhm'][idx]
         mask[idx] = True
      else:
         return -1,tab

      if plotfile is not None:
         ax.axhline(0.5, color='red')
         ax.axvline(fwhm/self.scale/2)

         # make the not-used profiles less prominent
         for i in range(len(mask)):
            if not mask[i]:
               ax.lines[2*i].set_alpha(0.05)
               ax.lines[2*i+1].set_alpha(0.1)
         ax.set_xlim(0, fwhm*10)
         fig.tight_layout()
         fig.savefig(plotfile)
      return fwhm,tab

   def plot_field(self, percent=99.):
      '''PLot the data as a field of view with LS stars plotted if loaded
      
         Args:
            percent(float):  the percentage of pixels for the colormap normalization
      '''

      fig = plt.figure()
      ax = fig.add_subplot(111, projection=self.wcs)
      norm = simple_norm(self.data, percent=percent) 
      ax.imshow(self.data, origin='lower', norm=norm, cmap='gray_r')

      if self.RAs is not None:
         ii,jj = self.wcs.wcs_world2pix(self.RAs, self.DEs, 0)
         ax.plot(ii, jj,'o', mfc='none', mec='red', ms=10)
         for k,lab in enumerate(self.objs):
            ax.text(ii[k]+10/self.scale, jj[k]+10/self.scale, str(self.objs[k]), ha='left',
                    color='red')
      
      return fig



   def doPhotometry(self, magins='MAGINS', stdcat='STDS.cat'):
      '''To be over-ridden by subclass'''
      pass

class PSFPhot(BasePhot):

   def __init__(self, ftsfile, tel='SWO', ins='NC', sigma=None, mask=None):
      super(PSFPhot,self).__init__(ftsfile, tel, ins, sigma, mask)

   def doPhotometry(self, magins='MAGINS', stdcat='STDS.cat'):
      '''Do the PSF photometry using the magins command.

      Returns:
         Astropy Table: Table of photometry. The following columns are included:
            xcenter,ycenter:  fit coordinates of the star
            msky:             mean sky value in sky aperture
            mskyerr:          uncertainty in msky
            flux[n],eflux[n]: flux and err in n'th aperture
            ap[n],ap[n]er:    magnitude in nth aperture
            filter:           filter of observation
            date:             JD of observation
            airmass:          airmass of observation
            objID:            identifyer of the LS star or SN (0)
            fits:             the FITS file of the observation
            flags:            bit-wise flags:  
                              1 = star outside frame
                              2 = aperture mag is NaN
                              4 = aperture mag error is NaN
                              8 = pixels masked out (saturated, e.g.)
            [F]mag,[F]err:    standard mag,error in filter F
      '''
      with tempfile.TemporaryDirectory() as tmpdir:
         catfile = os.path.join(tmpdir, 'cat')
         magfile = os.path.join(tmpdir, 'mags.txt')
         with open(catfile, 'w') as fout:
            for i in range(len(self.objs)):
               fout.write("{} {} {} {}\n".format(
                  self.object, self.objs[i], self.RAs[i], self.DEs[i]))
         com = "{} {} {} {} {}".format(magins, self.ftsfile, catfile, stdcat, 
               magfile)
         os.system(com)
         tab = ascii.read(magfile, names=['night','fits','tel','ins','filter', 
            'airmass','expt','ncombine','date','SN','OBJ','ra','dec','xc','yc',
            'mag1','merr1','mag2','merr2','mag3','merr3','mag4','merr4',
            'flux','msky','mskyer' ,'shart','chi','g1','g2','perr'])

      tab['msky'].info.format = "%.3f"
      tab['mskyer'].info.format = "%.3f"
      tab['date'].info.format = "%.2f"
      tab['airmass'].info.format = "%.3f"

      # Do some flags
      flags = zeros(len(tab), dtype=int)
      flags = where(tab['xc'] < 5, flags|1, flags)
      flags = where(tab['xc'] > self.data.shape[1]-5,flags|1,flags)
      flags = where(tab['yc'] < 5, flags|1, flags)
      flags = where(tab['yc'] > self.data.shape[0]-5,flags|1,flags)
      flags = where(tab['perr'] != "No_error",flags|2,flags)

      tab['flags'] = flags
      self.phot = tab

      return tab

class PSFPhot2(BasePhot):

   def __init__(self, ftsfile, tel='SWO', ins='NC', sigma=None, mask=None):
      super(PSFPhot2,self).__init__(ftsfile, tel, ins, sigma, mask)

   def ModelPSF(self, size=20, oversampling=4):
      '''Use the star catalog to make cutouts and model the PSF using
      photutil's PSFBuilder

      Args:
         size (int):  cutout size for the PSF data in arc-sec

      Returns:
         EPSFModel instance
      '''
      psize = int(size/self.scale)
      if psize % 2 == 0:  psize += 1

      # Determine the PSF approximate locations
      xc,yc = self.wcs.wcs_world2pix(self.RAs, self.DEs, 0)
      hsize = (size-1)/2
      gids = (xc > hsize) & (xc < (self.data.shape[1]-1-hsize)) &\
             (yc > hsize) & (yc < (self.data.shape[0]-1-hsize))
      tab = Table([xc[gids], yc[gids]], names=['x','y'])
      if len(tab) == 0:
         raise ValueError("Error:  All catalog stars are off-chip. Wrong field?")

      # extract the stars from the data
      mn,md,st = sigma_clipped_stats(self.data, sigma=2.)
      nddata = NDData(data=self.data - md)
      stars = extract_stars(nddata, tab, size=psize)

      # check to see if "saturated" (non-linear) pixels in cutouts
      bids = array([sometrue(s.data + md > self.datamax) for s in stars])
      self.saturated = bids
      if sometrue(bids):
         tab = tab[~bids]
      stars = extract_stars(nddata, tab, size=psize)

      # Make the EPSFBuilder. Seems that the quartic kernel has issues,
      # so we use the quartic
      eps_builder = EPSFBuilder(oversampling=oversampling, maxiters=10, 
          progress_bar=False, smoothing_kernel='quartic')
      epsf,fstars = eps_builder(stars)
      return epsf

   def doPhotometry(self, psfModel=None):
      '''Do the PSF photometry using the astropy photutils package.

      Args:
         psfModel:  An EPSFModel instance from self.ModelPSF or None, to use
                    a simple integrated Gaussian
      Returns:
         Astropy Table: Table of photometry. The following columns are included:
            xcenter,ycenter:  fit coordinates of the star
            msky:             mean sky value in sky aperture
            mskyerr:          uncertainty in msky
            flux[n],eflux[n]: flux and err in n'th aperture
            ap[n],ap[n]er:    magnitude in nth aperture
            filter:           filter of observation
            date:             JD of observation
            airmass:          airmass of observation
            objID:            identifyer of the LS star or SN (0)
            fits:             the FITS file of the observation
            flags:            bit-wise flags:  
                              1 = star outside frame
                              2 = aperture mag is NaN
                              4 = aperture mag error is NaN
                              8 = pixels masked out (saturated, e.g.)
            [F]mag,[F]err:    standard mag,error in filter F
      '''
      # Make sure we have what we need:
      if self.RAs is None or self.DEs is None:
         raise RuntimeError("No RA/DEC loaded, run loadObjCatalog first")

      # Instantiate the stuff we need
      mmm_bkg = MMMBackground()
      finder = FixedStarFinder(self.wcs, RAs=self.RAs, DECs=self.DEs, ids=self.objs)
      group = DAOGroup(10/self.scale)   # 10 arc-sec
      fitter = LevMarLSQFitter()
      if psfModel is None:
         psfModel = IntegratedGaussianPRF(sigma=1.0/self.scale)

      # Fit Size of the image:  10x10 arcsec
      pix = int(10.0/self.scale) 
      if pix % 2 == 0:  pix +=1
      fitshape = (pix,pix)

      psf = BasicPSFPhotometry(group_maker=group, bkg_estimator=mmm_bkg,
            psf_model=psfModel, fitshape=fitshape, finder=finder,
            fitter=fitter)
      self.tab = psf(self.data)
      self.resids = psf.get_residual_image()


class ApPhot(BasePhot):

   def __init__(self, ftsfile, tel='SWO', ins='NC', sigma=None, mask=None):
      '''Initialize this aperture photometry class with a tel/ins configuration
      and FITS file.
      
      Args: 
         ftsfile (str or FITS): The FITS file to deal with
         tel (str):  telescope code (e.g., SWO)
         ins (str):  instrument code (e.g., NC)
         sigma (str or FITS): The FITS file with error (noise) map
         mask (str or FITS): The optional FITS file with mask (True=Bad) 
      
      Returns:
         ApPhot instance.
      '''
      super(ApPhot,self).__init__(ftsfile, tel, ins, sigma, mask)
      self.apps = []
      self.skyap = None

   def centroid(self, cutoutsize=20, method='1dg'):
      '''Given data and a wcs, centroid the catalog objects. Returns
      as a SkyCoord object.

      Args:
         boxsize (int):  size of the extraction box
         method (str):   centroiding method to use (see photutils docs)

      Returns:
         astropy.SkyCoords:   Coordinates of the objects
      '''
      if self.RAs is None:
         raise ValueError("You need to load a catalog first")
      # re-centroid the positions:
      xs,ys = self.wcs.wcs_world2pix(self.RAs, self.DEs, 0)
      flags = []; xout = []; yout = []
      if cutoutsize % 2 == 0: cutoutsize += 1   # make sure odd
      interv = cutoutsize//2                    # integer division!!!
 
      for i in range(len(xs)):
         if not (interv < xs[i] < self.data.shape[1]-interv and\
               interv < ys[i] < self.data.shape[0]-interv):
            xout.append(xs[i])
            yout.append(ys[i])
            flags.append(1)    # outside data array
         else:
            x = int(xs[i])
            y = int(ys[i])
            sdata = self.data[y-interv:y+interv+1, x-interv:x+interv+1]
            mn,md,st = sigma_clipped_stats(sdata, sigma=2.)
            if method == 'com':
               xx,yy = centroid_com(sdata - md)
            elif method == '1dg':
               xx,yy = centroid_1dg(sdata - md)
            elif method == '2dg':
               xx,yy = centroid_2dg(sdata - md)
            elif method == 'quad':
               xx,yy = centroid_quadratic(sdata - md)
            if not (0 < xx < cutoutsize and 0 < yy < cutoutsize):
               xout.append(xs[i])
               yout.append(ys[i])
               flags.append(2)
            else:
               xout.append(x - interv + xx)
               yout.append(y - interv + yy)
               flags.append(0)
      ra,dec = self.wcs.wcs_pix2world(array(xout), array(yout), 0)
      self.centroids = SkyCoord(ra*u.deg, dec*u.deg)


   def makeApertures(self, appsizes=[3,5,7], sky_in=9, sky_out=11,
         boxsize=20):
      '''Create the aperture objects.
      
      Args:
         appsizes (list of floats): list of aperture radii in arc-sec.
         sky_in (float):  inner aperture for sky annulus
         sky_out (float):  outer aperture for sky annulus
         
      Returns:
         None
         
      Effects:
         creates the aperture objetcs self.apps
      '''
      if self.centroids is None:
         with catch_warnings():
            simplefilter('ignore')
            self.centroid(cutoutsize=boxsize)
      
      for apsize in appsizes:
         self.apps.append(SkyCircularAperture(self.centroids, 
            r=float(apsize)*u.arcsec).to_pixel(self.wcs))
      self.apps.append(SkyCircularAnnulus(self.centroids, 
            r_in=float(sky_in)*u.arcsec, 
            r_out=float(sky_out)*u.arcsec).to_pixel(self.wcs))

   def estimateSky(self):
      '''Estimate the sky levels in the apertures.
      
      Returns:
         (sky,skyerr):  arrays of sky values and errors
      '''
      if not self.apps:
         self.makeApertures()
      annmasks = self.apps[-1].to_mask(method='center')
      skies = []
      eskies = []
      for mask in annmasks:
         anndata = mask.multiply(self.data)
         if anndata is None:
            skies.append(nan)
            eskies.append(nan)
            continue
         anndata = anndata[mask.data > 0]
         _,md,st = sigma_clipped_stats(anndata, sigma=3.0)
         skies.append(md)
         eskies.append(st/sqrt(anndata.shape[0]))
      return array(skies), array(eskies)

   def doPhotometry(self):
      '''Do the actual photometry.

      Returns:
         Astropy Table: Table of photometry. The following columns are included:
            xcenter,ycenter:  fit coordinates of the star
            msky:             mean sky value in sky aperture
            mskyerr:          uncertainty in msky
            flux[n],eflux[n]: flux and err in n'th aperture
            ap[n],ap[n]er:    magnitude in nth aperture
            filter:           filter of observation
            date:             JD of observation
            airmass:          airmass of observation
            objID:            identifyer of the LS star or SN (0)
            fits:             the FITS file of the observation
            flags:            bit-wise flags:  
                              1 = star outside frame
                              2 = aperture mag is NaN
                              4 = aperture mag error is NaN
                              8 = pixels masked out (saturated, e.g.)
            [F]mag,[F]err:    standard mag,error in filter F
      '''
      if not self.apps:
         self.makeApertures()
      phot_table = aperture_photometry(self.data, self.apps[0:-1],
            error=self.error, mask=self.mask)
      # Figure out the sky
      msky,esky = self.estimateSky()

      phot_table['msky'] = msky
      phot_table['mskyer'] = esky
      phot_table['msky'].info.format = "%.3f"
      phot_table['mskyer'].info.format = "%.3f"

      for i in range(len(self.apps)-1):
         key = 'aperture_sum_{}'.format(i)
         ekey = 'aperture_sum_err_{}'.format(i)
         flux = phot_table[key] -  msky*self.apps[i].area
         eflux = sqrt(power(esky*self.apps[i].area,2) +
                        power(phot_table[ekey],2))
         phot_table.remove_column(key)
         phot_table.remove_column(ekey)
         fkey = 'flux{}'.format(i)
         efkey = 'eflux{}'.format(i)
         phot_table[fkey] = flux
         phot_table[fkey].info.format = "%.3f"
         phot_table[efkey] = eflux
         phot_table[efkey].info.format = "%.3f"
         akey = 'ap{}'.format(i)
         eakey = 'ap{}er'.format(i)
         phot_table[akey] = -2.5*log10(phot_table[fkey]/self.exposure) + 30
         phot_table[akey].info.format = "%.3f"
         phot_table[eakey] = phot_table[efkey]/phot_table[fkey]*1.087
         phot_table[eakey].info.format = "%.3f"
      phot_table['filter'] = self.filter
      phot_table['date'] = self.date
      phot_table['date'].info.format = "%.2f"
      phot_table['airmass'] = self.airmass
      phot_table['airmass'].info.format = "%.3f"
      phot_table['OBJ'] = self.objs
      phot_table['fits'] = self.ftsfile

      # Do some flags
      flags = zeros(len(phot_table), dtype=int)
      flags = where(phot_table['xcenter'].value < 5, flags|1, flags)
      flags = where(phot_table['xcenter'].value > self.data.shape[1]-5, flags|1,flags)
      flags = where(phot_table['ycenter'].value < 5, flags|1, flags)
      flags = where(phot_table['ycenter'].value > self.data.shape[0]-5, flags|1,flags)
      for i in range(len(self.apps)-1):
         flags = where(isnan(phot_table['ap{}'.format(i)]),
                       flags | 2, flags)
         flags = where(isnan(phot_table['ap{}er'.format(i)]),
                       flags | 4, flags)

      # Check if masked pixels occurred in the apertures
      ap_masks = [ap.to_mask() for ap in self.apps[0:-1]]
      # indexed by [ap,obj]
      maps = [[sometrue(am.multiply(self.mask)) for am in ams] \
               for ams in ap_masks]
      flags = where(sometrue(maps, axis=0), flags | 8, flags)
      phot_table['flags'] = flags
      self.phot = phot_table

      return phot_table

   def plotCutOuts(self, xcols=4, ycols=4, apindex=-2):
      if self.centroids is None:
         self.centroid()
      if not self.apps:
         self.makeApertures()

      figrat = ycols/xcols

      NplotsPage = xcols*ycols
      Npages = len(self.centroids)//NplotsPage
      if not (len(self.centroids) % NplotsPage == 0):  Npages += 1
      
      norm = simple_norm(self.data, 'linear', percent=99.)

      bbs = self.apps[-1].bbox

      for p in range(Npages):
         fig,ax = plt.subplots(xcols, ycols, figsize=(8,8*figrat))
         plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0,
               wspace=0)
         ax = ax.ravel()
         for i,idx in enumerate(range(p*NplotsPage,(1+p)*NplotsPage)):
            if idx > len(self.centroids) - 1: break
            bb = bbs[idx]
            ax[i].imshow(self.data[bb.slices], origin='lower', norm=norm)
            orig = (bb.ixmin, bb.iymin)
            self.apps[apindex].plot(origin=orig, indices=idx, color='white', 
                  axes=ax[i])
            self.apps[-1].plot(origin=orig, indices=idx, color='red', 
                  axes=ax[i], lw=1)
            ax[i].text(0.9, 0.9, str(self.objs[idx]), va='top', ha='right',
                  transform=ax[i].transAxes, color='white')
         for i in range(NplotsPage):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
         #fig.tight_layout()
         fig.savefig(self.ftsfile+"_cutout{}.png".format(p))

def NN1(phot, std_key, inst_key='ap2', sigma=3, niter=3, fthresh=0.3):
   '''Construct the N(N-1) combinations of differences between the instrumental
   photometry and standard photometry. The idea is that, for star i and star
   j, the differece m_inst(i) - m_inst(j) should be close to
   m_stand(i) - m_stand(j).'''

   idiff = phot[inst_key][newaxis,:] - phot[inst_key][:,newaxis]
   sdiff = phot[std_key][newaxis,:] - phot[std_key][:,newaxis]
   oids = indices(idiff.shape)

   # Now we are only interested in the upper-triangular portion by symmetry
   idiff = triu(idiff).ravel()
   sdiff = triu(sdiff).ravel()
   oid1 = triu(oids[0]).ravel()
   oid2 = triu(oids[1]).ravel()

   # Now raval() and take out the zeros lower triangular and diagonal as well
   # as the supernova object
   gids = greater(oid1, 0) & greater(oid2,0)
   gids = gids*~equal(oid1,oid2)   # remove the trivial cases
   idiff = idiff[gids]
   sdiff = sdiff[gids]
   oid1 = oid1[gids]
   oid2 = oid2[gids]

   bids = isnan(sdiff-idiff)   # should give all False
   omit = []
   thresh = int(len(phot)*fthresh)
   # Now we look for outliers
   for i in range(niter):
      mad = 1.5*median(absolute(sdiff-idiff)[~bids])
      bids = greater(absolute(sdiff-idiff), sigma*mad)
      # These have the ids and counts of objects that produce large outliers
      u1,c1 = unique(oid1[bids], return_counts=True)
      u2,c2 = unique(oid1[bids], return_counts=True)

      # Because 1 outlier can cause N discrepancies, we want counts that
      # exceed some fraction of the whole (fthresh).
      omit = omit + list(u1[c1 > thresh]) + list(u2[c2 > thresh])
      omit = list(set(omit))

   return idiff,sdiff,oid1,oid2,omit

def compute_zpt(phot, std_key, stderr_key, inst_key='ap2', ierr_key='ap2er',
      magmin=15, magmax=20, objkey='objID', zpins=30, plot=None, emagmax=0.5,
      use_pymc=False):
   '''Given a table of photometry, determine the zero-point as a weighted
   offset between the standard photometry and instrumental photometry.

   Args:
      phot (astropy.Table):  Table of aperture photometry output from ApPhot
      std_key (str):  the standard magnitude key in the table.
      stderr_key (str):  the standard magnitude errors in the table
      inst_key (str): The key for the aperture magnitudes in the table
      ierr_key (str): The instrumental magnitude error in the table
      magmin (float):  minimum standard magnitude mag < magmin are rejected
      magmax (float0:  maximum standard magnitude mag > magmax are rejected
      objkey (str):  The object designation column in the table
      zpins (float):  The approximate zero-point initially used to compute
                     instrumental magntiudes
      plot (None|str|Axes):  Plot the zp and ap-corr residuals? if not None
                     and a string, output plot to file. If Axes, plot into that
                     Axes instance.
      emagmax (float): Maximum error in std mag or inst mag to consider
      use_pymc (bool): If true, use pymc3 to compute a more robust error in zp.

   Returns:
      (zp,ezp,flags,mesg)
      zp (float):  the zero-point
      ezp (float):  the error in zero-point
      flags (array):  integers denoting issues: 1- NN rejection
                                                2- min/max rejection
                                                4- sigma-clipped
                                                8- error too large
      mesg (str):  Useful message

      These are set to None,None,None,error-mesg if an error occurs

   Effects:
      if plot is a filename, a plot is made with that filename
   '''
   flags = zeros((len(phot),), dtype=int)

   # First, we're going to look for objects with inconsistent differentials
   idiff,sdiff,oid1,oid2,omit = NN1(phot, std_key, inst_key)
   flags[omit] += 1

   gids = greater(phot[objkey], 0)        # remove SN
   gids = gids*equal(phot['flags'], 0)    # Get rid of photometry prolems
   gids = gids*between(phot[std_key], magmin, magmax)
   gids = gids*less(phot[stderr_key], emagmax)*less(phot[ierr_key],emagmax)
   flags[greater(phot[stderr_key],emagmax)+greater(phot[ierr_key],emagmax)] += 8
   flags[~between(phot[std_key], magmin, magmax)] += 2
   gids[omit] = False   # get rid of objects found above
   if not sometrue(gids):
      mesg = "Not enough good photometric points to solve for zp"
      return None,None,None,mesg
   diffs = phot[std_key]- phot[inst_key]
   mn,md,st = sigma_clipped_stats(diffs[gids], sigma=3)

   # throw out 5-sigma outliers with respect to MAD
   mad = 1.5*median(absolute(diffs - md))
   gids = gids*less(absolute(diffs - md), 5*mad)
   flags[~less(absolute(diffs - md), 5*mad)] += 4
   if not sometrue(gids):
      mesg = "Too many outliers in the photometry, can't solve for zp"
      return None,None,None,mesg

   if use_pymc and pymc is not None:
      mvar = array(phot[gids][ierr_key]**2 + phot[gids][stderr_key]**2)
      with pymc.Model() as model:
         zp = pymc.Uniform('zp', -10, 10)
         sigma = pymc.Uniform('sigma', 0, 5)
         sigtot = sqrt(mvar + sigma**2)
         mod = pymc.Normal('obs', mu=zp, sd=sigtot, observed=array(diffs[gids]))
         trace = pymc.sample(5000, chains=4, cores=4, tune=1000, 
               progressbar=False)
      zp = median(trace['zp'])
      ezp = std(trace['zp'])
   else:
      # Weight by inverse variance
      wts = power(phot[ierr_key]**2 + phot[stderr_key]**2,-1)*gids

      # zpins is used internall in photometry code as arbitrary zero-point
      zp = sum(diffs*wts)/sum(wts) + zpins
      ezp = sqrt(1.0/sum(wts))

   if plot is not None:
      if isinstance(plot, plt.Axes):
         ax = plot
         fig = None
      else:
         fig,ax = plt.subplots()
      ax.errorbar(phot[std_key], diffs + zpins - zp, fmt='o', 
            xerr=phot[stderr_key], 
            yerr=sqrt(phot[ierr_key]**2 + phot[stderr_key]**2))
      ax.plot(phot[std_key][~gids], diffs[~gids] + zpins - zp, 'o', mfc='red',
            zorder=100)
      ax.axhline(0, color='k')
      if use_pymc and pymc is not None:
         ax.axhline(median(trace['sigma']), color='k', linestyle='--')
         ax.axhline(median(-trace['sigma']), color='k', linestyle='--')

      ax.set_xlim(12,20)
      ax.set_ylim(-1,1)
      ax.set_xlabel('m(std)')
      ax.set_ylabel('m(std) - m(ins)')
      if fig is not None:
         fig.savefig(plot)

   return zp,ezp,flags,"ok"


from photutils.detection import StarFinderBase
class FixedStarFinder(StarFinderBase):

   def __init__(self, wcs, RAs=None, DECs=None, ids=None, catalog=None, radius=12):

      if RAs is not None and DECs is not None and ids is not None:
         self.tab = Table([ids, RAs, DECs], names=['id','RA','DEC'])
      elif catalog is not None:
         self.tab = ascii.read(catalog)
         if 'RA' not in self.tab.colnames or 'DEC' not in self.tab.colnames \
             or 'id' not in tab.colnames:
            raise ValueError("Error:  the input catalog must have RA, DEC, and "\
                  "id columns")
      else:
         raise ValueError("Error: must specify RAs,DECs,ids or catalog")

      self.wcs = wcs
      self.radius = radius
      
   def find_stars(self, data, mask=None):
      i,j = self.wcs.wcs_world2pix(self.tab['RA'], self.tab['DEC'], 0)
      self.tab['xcentroid'] = i
      self.tab['ycentroid'] = j
      self.tab['flux'] = 0
      for idx in range(len(i)):
         i0 = int(i[idx]-self.radius)
         j0 = int(j[idx]-self.radius)
         i1 = int(i[idx]+self.radius)
         j1 = int(j[idx]+self.radius)
         if i0 < 0 or j0 < 0 or i1 > data.shape[1]-1 or j1 > data.shape[0]-1:
            # Too close to edges
            self.tab['xcentroid'][idx] = -1
            self.tab['ycentroid'][idx] = -1
            self.tab['flux'][idx] = 0
         else:
            cutout = data[j0:j1,i0:i1]
            self.tab[idx]['flux'] = sum(cutout.ravel())
      self.tab = self.tab[self.tab['xcentroid'] > 0]

      return(self.tab)

'''A module for performing photometry. Uses photutils which, at this point,
seems to only do aperture photometry reliably.'''

from .config import config
from .tel_specs import getTelIns

from astropy.stats import mad_std, sigma_clipped_stats
import astropy.units as u
from astropy.io import ascii,fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time

from photutils import make_source_mask
from photutils import SkyCircularAperture, SkyCircularAnnulus
from photutils import aperture_photometry
from photutils.centroids import centroid_com,centroid_1dg

from matplotlib import pyplot as plt
from astropy.visualization import simple_norm

from warnings import warn
import sys,os
from numpy import *
from .npextras import between
from astropy.table import vstack

try:
   import pymc3 as pymc
except:
   pymc = None

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


class ApPhot:

   def __init__(self, ftsfile, tel='SWO', ins='NC', sigma=None):
      '''Initialize this aperture photometry class with a tel/ins configuration
      and FITS file.
      
      Args: 
         ftsfile (str or FITS): The FITS file to deal with
         tel (str):  telescope code (e.g., SWO)
         ins (str):  instrument code (e.g., NC)
         sigma (str or FITS): The FITS file with error (noise) map
      
      Returns:
         ApPhot instance.
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
      self.wcs = WCS(self.head)

      self.RAs = None
      self.DEs = None
 
      self.parse_header()
      if sigma is None:
         self._makeErrorMap()
      else:
         if isinstance(sigma, str):
            sigma = fits.open(sigma)
         self.error = sigma[0].data

      self.centroids = None
      self.apps = []
      self.skyap = None
      self.phot = None

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
                  warn("Header keyword {}, not found, using fallback={}".format(
                     fitskey, fallback))
                  return fallback
               raise KeyError("header keyword {} not found".format(fitskey))
            return self.head[fitskey.upper()]
         else:
            return val
      else:
         if fallback is not None:
            warn("Couldn't figure out value for {}, using fallback={}".format(
               key, fallback))
            return fallback
         else:
            raise KeyError("Option {} not found in data section".format(key))

   def parse_header(self):
      '''Parse the FITS header for information we'll need, falling back on
      defaults or values from the config file.'''
      self.exposure = self._parse_key("exposure")
      self.filter = self._parse_key("filter")
      self.date = self._parse_key("date")
      self.gain = self._parse_key("gain", fallback=1.0)
      self.ncombine = self._parse_key("ncombine", fallback=1)
      self.rdnoise = self._parse_key("rnoise", fallback=0.0)
      self.airmass = self._parse_key("airmass", fallback=1.0)

   def _makeErrorMap(self):
      '''Compute the error in the data based on gain, readnoise, etc.'''
      # effective gain:
      egain = self.gain*self.ncombine
      self.error = where(self.data > 0,
            sqrt(self.data/egain + self.rdnoise**2/self.gain**2),
            self.rdnoise/self.gain)

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

   def centroid(self, boxsize=20):
      '''Given data and a wcs, centroid the catalog objects. Returns
      as a SkyCoord object.'''
      if self.RAs is None:
         raise ValueError("You need to load a catalog first")
      # re-centroid the positions:
      i,j = self.wcs.wcs_world2pix(self.RAs, self.DEs, 0)
      ii,jj,flag = recenter(i, j, self.data, cutoutsize=boxsize)
      ra,dec = self.wcs.wcs_pix2world(ii, jj, 0)
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
         self.centroid(boxsize=boxsize)
      
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
      '''Do the actual photometry.'''
      if not self.apps:
         self.makeApertures()
      phot_table = aperture_photometry(self.data, self.apps[0:-1],
            error=self.error)
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
      flags = where(phot_table['ycenter'].value < 5, flags|11, flags)
      flags = where(phot_table['ycenter'].value > self.data.shape[0]-5, flags|1,flags)
      for i in range(len(self.apps)-1):
         flags = where(isnan(phot_table['ap{}'.format(i)]),
                       flags | 2, flags)
         flags = where(isnan(phot_table['ap{}er'.format(i)]),
                       flags | 4, flags)
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

   gids = greater(phot[objkey], 0)
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
      if use_pymc:
         ax.axhline(median(trace['sigma']), color='k', linestyle='--')
         ax.axhline(median(-trace['sigma']), color='k', linestyle='--')

      ax.set_xlim(12,20)
      ax.set_ylim(-1,1)
      ax.set_xlabel('m(std)')
      ax.set_ylabel('m(std) - m(ins)')
      if fig is not None:
         fig.savefig(plot)

   return zp,ezp,flags,"ok"


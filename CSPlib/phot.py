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
from astropy.table import vstack

def recenter(xs, ys, data, cutoutsize=40):
   '''Given initial guesses for star positions, re-center.'''
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
         xx,yy = centroid_1dg(sdata - md)
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

   def __init__(self, ftsfile, tel='SWO', ins='NC'):
      '''Initialize this aperture photometry class with a tel/ins configuration
      and FITS file.
      
      Args: 
         ftsfile (str or FITS): The FITS file to deal with
         tel (str):  telescope code (e.g., SWO)
         ins (str):  instrument code (e.g., NC)
      
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
      self._makeErrorMap()

      self.centroids = None
      self.apps = []
      self.skyap = None

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

      bbs = self.apps[-1].bounding_boxes

      for p in range(Npages):
         fig,ax = plt.subplots(xcols, ycols, figsize=(8,8*figrat))
         ax = ax.ravel()
         for i,idx in enumerate(range(p*NplotsPage,(1+p)*NplotsPage)):
            if idx > len(self.centroids) - 1: break
            bb = bbs[idx]
            ax[i].imshow(self.data[bb.slices], origin='lower', norm=norm)
            orig = (bb.ixmin, bb.iymin)
            self.apps[apindex].plot(origin=orig, indices=idx, color='white', 
                  ax=ax[i])
            self.apps[-1].plot(origin=orig, indices=idx, color='red', ax=ax[i],
                  lw=1)
            ax[i].text(0.9, 0.9, str(self.objs[idx]), va='top', ha='right',
                  transform=ax[i].transAxes, color='white')
         for i in range(NplotsPage):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
         fig.tight_layout()
         fig.savefig(self.ftsfile+"_cutout{}.png".format(p))


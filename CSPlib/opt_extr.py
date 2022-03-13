'''Module opt_extr, optimal extraction photometry.  Shamelessly copied from T.
Naylor's fortran90 code.

See his documentation on the use of the software.  However, one note on
coordinate systems:  positions are specified as fractional pixels with the
lower-left pixel being (0,0), NOT (1,1).  So beware if you use sextractor or
IRAF to make your coordinate lists, they have the lower-left pixel set to
(1,1).

Note to python users:  the usual IRAF way of interpreting (x,y) is that x is
the horizontal and y is the vertical coordinate.  In terms of storage, x, is
the fastest varying coordinate, y the slowest.  However, in terms of Numeric
arrays, this means that if (x,y) are the coordinates, then the data is accessed
as data[y,x].

Here follows original comments:


      This is the module that carries out the optimal extraction.

      Version 3.0

      There are two subroutines designed to be used by the programmer creating
      a programme to carry out optimal extraction.  psf_calc will calculate
      the model PSF, and opt_calc will carry out the optimal photometry.

      By Tim Naylor

      Modifications.
      Version 1.1 
        Has the subroutines skyhst and skymod.
      Version 1.2 
        Allows flux_aper to return the peak counts in the aperture.
      Version 2.0
        Allows the position angle of the Gaussian to run free.
      Version 2.1
        Bug cured whereby position angle was running free even after
        the shape parameters had been fixed.
      Version 2.2
        More comments to aid porting added.
      Version 2.3
        Returns an estimate of the error in the position.
      Version 2.4
        Negative glitches foul up the noise model as data-sky becomes
        negative, which can make skynos*skynos + (data-sky)/adu negative,
        and hence ask for the square root of a naegative number.  Obviously
        photometry around such a pixel is junk, but as we don't want the
        program to crash, it will set the noise to be the sky noise.
      Version 2.5
        Skyfit now spots if there are no good pixels available in the sky box.
      Version 2.6
        Minor Changes so that it compiles with the NAG Fortran 95
        complier, as well as the DEC Fortran 90 compiler.
      Version 2.7
        Bug cured whereby the rotation of the Gaussian fitted to the PSF
        star was limited to 0 -> 45 degrees, when it should be -45 -> +45.
      Version 2.8
        Minor change to opt_extr to speed up calculation of var for sky
          limited case.
        Cured a bug which crashed the program if the companion star was off
          the frame edge.
        Checks for high skewness in the sky histogram.
        Uses iterative sigma clipping to make initial sky and sigma estimates
          in skyfit.
        Lee Howells' fixes for a couple of bugs added.
        GFIT arrays normalised before going into curfit to over numerical
          overflows.
      Version 3.0
        General tidy-up of code, including changing many subroutine arguments.
        Introduction of pixel flags.
        Flags changed to characters rather than integers.
      Version 3.1
        Two bugs found that only realy show up when fitting a negative sky.
        The derivative for the position of the peak of the skewed Gaussian 
        was not calculated properly when the sky was negative, and the 
        clipped mean wasn't done properly either.
'''
import matplotlib
matplotlib.use('Agg')
import sys,os,string
import numpy as np
from .tel_specs import getTelIns
from astropy.modeling import models,fitting
from astropy.stats import sigma_clipped_stats
from astropy.io import fits,ascii
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, scott_bin_width
from .npextras import between
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)

## Fit histograms to determine sky levels
def skwfit(xdata, ydata, npts, skynos, debug=False):
   '''returns (icurf, skyerr, skynos, b_chi, par)

   The function fits the first npts points of the function ydata(xdata)
   with a skewed Gaussian, and then returns the fwhm and the peak
   position (skwfit).  icurf is the curfit status flag or set to -200
   if the fwhm became too small to fit with the binning given.  skynos is
   the sigma of the sky, on input it should be an estimate of this.'''

   # Set value of maximum.
   ymax = max(ydata)
   ixmax = np.argmax(ydata)

   # Set fwhm, to best guess based on sky deviation, but not less than
   # the binning.
   skymod = skgauss(beta=0, x0=xdata[ixmax], y0=1.0, 
         fwhm=max(skynos/0.416277, xdata[1]-xdata[0]))
   skymod.beta.fixed = True
   skymod.x0.fixed = True
   skymod.y0.fixed = True
   skymod.fwhm.min = 0.0
   
   # Divide fit by ymax (to stop curfit overflows), and set weighting.
   if (debug):  self.log('@ Dividing y values by {}'.self.log(ymax))
   w = ydata*0.0 + ymax**2
   w = np.where(ydata > 0.0, ymax**2/ydata, w)
   w = np.where(ydata < 0.1*ymax, 0.0, w)
   #for i in range(len(ydata)):
   #   w[i] = ymax**2
   #   if (ydata[i] > 0.0):  w[i] = ymax*ymax/ydata[i]
   #   if (ydata[i] < 0.1*ymax):  w[i] = 0.0
   ydata = 1.0*ydata/ymax
   # our LM routine expects sigma, not variance
   if debug:
      self.log('@ Parameters; skew, X(Y max), scaled Y max, fwhm.')
      self.log('@           {}, {}, {}, {}'.format(*skymod.parameters))

   fitter = fitting.LevMarLSQFitter()
   fit = fitter(skymod, xdata, ydata, weights=w)
   if fitter.fit_info['ierr'] not in [1,2,3,4]:
      if debug: self.log('@ Warning, fit (1) failed {} {}'.format(
         fitter.fit_info['ierr'], fitter.fit_info['message']))
      icurf = -100
   else:
      icurf = 0

   # If the sigma has gone smaller than the histogram step size
   # it is dangerous to continue (the fit may crash).
   if debug:  self.log("FITSKW:  fwhm first pas is {}".format(out.params[3]))
   if (fit.fwhm < (xdata[-1]-xdata[1])/float(npts-1)):
      icurf = -200
      if (debug):  self.log('Binning was {}'.format(
         (xdata[-1]-xdata[1])/float(npts-1)))
   else:
      # Free the normalisation.
      fit.y0.fixed = False
      fit.x0.fixed = False
      if (debug):
         self.log('@ ** Now fitting with free position and normalisation.')
         self.log('@ Parameters; skew, X(Y max), scaled Y max, fwhm.')
         self.log('@  parinfo = {} {} {} {}'.format(*fit.parameters))
      fit = fitter(fit, xdata, ydata, weights=w)
      if fitter.fit_info['ierr'] not in [1,2,3,4]:
         if debug: self.log('@ Warning, fit (2) failed {} {}'.format(
            fitter.fit_info['ierr'], fitter.fit_info['message']))
         icurf = -100
         # reset everything, and see if it can fit with non zero sky_par[0]
         fit.x0.value = xdata[ixmax]
         fit.y0.value = 1.0
         fit.fwhm.value = max(skynos/0.416277, xdata[1]-xdata[0])
      else:
         icurf = 0

      # If the sigma has gone smaller than the histogram step size
      # it is dangerous to continue (the fit may crash).
      if debug: self.log("FITSKW:  fwhm second pas is {}".format(fit.fwhm.value))
      if debug: self.log("xdata[-1]={} xdata[1]={} npts={}".format(
         xdata[-1],xdata[1],npts))
      if (fit.fwhm.value < (xdata[-1]-xdata[1])/float(npts-1)):
         icurf = -200
         if (debug):  self.log('Binning was {}'.format(
            (xdata[-1]-xdata[1])/float(npts-1)))
      else:
         # Now add a little cockeyedness, make it similar to the setp used
         # for calculating the derivatives in curfit.
         fit.beta.fixed=False
         fit.beta.min = 0.0
         fit.beta.value = 0.05   # get it off the boundary
         if (debug):
            self.log('@ ** Now fitting with free skew.')
         fit = fitter(fit, xdata, ydata, weights=w)
         if fitter.fit_info['ierr'] not in [1,2,3,4]:
            if debug: self.log('@ Warning, fit (3) failed {} {}'.format(
               fitter.fit_info['ierr'], fitter.fit_info['message']))
            icurf = -100
         else:
            icurf = 0
   if (debug):   self.log('@ Errors were {}'.format(
      np.sqrt(np.diag(fitter.fit_info['cov_x']))))
   if (debug):   self.log('@ final params {}'.format(fit.parameters))
   
   skwfit=fit.x0.value
   b_chi = np.sum(np.power(fitter.fit_info['fvec'],2))/(npts-4)
   skynos=0.416277*fit.fwhm.value
   if fitter.fit_info['cov_x'] is not None:
      perror =  np.sqrt(np.diag(fitter.fit_info['cov_x']))
      skyerr=perror[1]
   else:
      skyerr = -1
   if (debug):   self.log('@  skyerr={}  skynos={}'.format(skyerr,skynos))
   return (icurf, skyerr, skynos, b_chi,fit.parameters)
   
@models.custom_model
def skgauss(x, beta=1, x0=0, y0=0, fwhm=10):
   '''Returns a skewed Gaussian.
   x:  x values of histogram
   beta: skew
   x0: center
   y0: amplitude
   fwhm: full-width at half-maximum'''

   gauss = y0*np.exp(-np.log(2.0)*(np.power(2.0*(x-x0)/fwhm,2) ))
   if abs(beta)<0.001:
      # Near enough a normal Gaussian
      return(gauss)

   left_ids = np.nonzero(np.less(x-x0, 0));   # Where just a gaussian
   # The function used here is a normal skewed Gaussian, except that
   # we use the normal definition of fwhm and multiply it by
   # sinh(beta)/(1.0 - exp(-1.0*beta)).  This means the left 
   # hand side of the curve is the same as a Gaussian with that FWHM.
   gamma=fwhm*beta/(1.0 - np.exp(-1.0*beta))
   # For the normal normalisation this would be
   # gamma=fwhm*beta/sinh(beta)
   check=2.0*beta*(x-x0)/gamma
   check2 = np.where(check <= -1.0, 0.0, check)
   fun = np.where(check <= -1.0, 0.0, 
         y0*np.exp(-np.log(2.0)*np.power(np.log(1.0+check2)/beta, 2)))
   np.put(fun, left_ids, gauss)
   return fun 

# Note that the variables here are not the standard notation, but make them
# nearly the same as Gaussian2d, which makes things programatically 
# Convenient
@models.custom_model
def Moffat2DEl(x,y,amplitude=1,x_mean=0,y_mean=0,x_stddev=1,y_stddev=1,theta=0,
      beta=1):
   xx = 1.0*x - x_mean
   yy = 1.0*y - y_mean
   r = np.sqrt(xx*xx + yy*yy)
   # If r is zero then the function will come out O.K. whatever theta is.
   # Otherwise find theta, getting the quadrant right.
   arg = np.where(r > 0.0, xx/r, 0.0)
   theta0 = np.where(r > 0.0, np.sign(yy)*np.arccos(arg), 0.0)
   # And make the argument to the exponential in those terms.
   work = np.power(r,2)*(np.power(np.cos(theta0+theta)/x_stddev,2) \
         + np.power(np.sin(theta0+theta)/y_stddev, 2) )
   moffat = amplitude*np.power((1+work), -1.0*beta)
   return moffat

def makeGaussian2d(a_par):
   '''Given a_par, make a model with the right parameters'''
   return models.Gaussian2D(a_par[3],a_par[4], a_par[5], a_par[0], a_par[1],
         a_par[2])

def makeMoffat2DEl(a_par):
   '''Given a_par, make a model with the right parameters'''
   return Moffat2DEl(a_par[3],a_par[4], a_par[5], a_par[0], a_par[1],
         a_par[2], a_par[6])


class OptExtrPhot:

   def __init__(self, ftsfile, tel='SWO', ins='NC', debug=False, 
         logf=sys.stdout):
      '''Initialize this optimal extraction photometry class with a tel/ins 
      configuration and FITS file.
      
      Args: 
         ftsfile (str or FITS): The FITS file to deal with
         tel (str):  telescope code (e.g., SWO)
         ins (str):  instrument code (e.g., NC)
      
      Returns:
         OptExtrPhot instance.
      '''
      self.cfg = getTelIns(tel,ins)
      self.logf = logf
      self.debug = debug
      if not os.path.isfile(ftsfile):
         raise ValueError("Error:  not such file {}".format(ftsfile))
      if isinstance(ftsfile, str):
         ftsobj = fits.open(ftsfile)
         self.ftsfile = ftsfile
         closeit=True
      else:
         ftsobj = ftsfile
         self.ftsfile = None
         closeit=False

      self.head = ftsobj[0].header
      self.data = ftsobj[0].data
      if closeit:  ftsobj.close()
      self.high = (self.data.shape[1],self.data.shape[0])
      self.low = (0,0)
      self.wcs = WCS(self.head)

      self.RAs = None
      self.DEs = None
 
      self.parse_header()
      #self.datamax = self.cfg.get('datamax',65000)
      self._makeErrorMap()

      self.centroids = None

      self.profile = 'moffat'
      #self.apps = []
      #self.skyap = None

   def log(self,message):
      self.logf.write(message+"\n")

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
                  self.log("Header keyword {}, not found, using fallback={}".\
                        format( fitskey, fallback))
                  return fallback
               raise KeyError("header keyword {} not found".format(fitskey))
            return self.head[fitskey.upper()]
         else:
            return val
      else:
         if fallback is not None:
            warnings.warn(
               "Couldn't figure out value for {}, using fallback={}".format(
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
      self.obj_name = self._parse_key("object", fallback=None)
      self.scale = self._parse_key("scale", fallback=1.0)
      self.datamax = self._parse_key("datamax", fallback=6.5e4)

   def _makeErrorMap(self):
      '''Compute the error in the data based on gain, readnoise, etc.'''
      # effective gain:
      egain = self.gain*self.ncombine
      self.error = np.where(self.data > 0,
            np.sqrt(self.data/egain + self.rdnoise**2/self.gain**2),
            self.rdnoise/self.gain)
      self.pix_flg = np.greater(self.error, 0)

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

      self.xpsf,self.ypsf = self.wcs.wcs_world2pix(self.RAs,self.DEs,0)

      # Only take stars that are on the chip
      gids = between(self.xpsf,10,self.data.shape[1]-10) & \
             between(self.ypsf,10,self.data.shape[0]-10)
      self.xpsf = self.xpsf[gids]
      self.ypsf = self.ypsf[gids]
      self.objs = self.objs[gids]

   def psf_calc(self, dpsf, fwhm, noise=None, plotfile=None): 
      '''returns   (shape_par, ipsf, nfit)
   
      This subroutine calculates the point spread function.
   
      The idea is that this subroutine is given a list of PSF stars.  It
      will fit the first 49 it finds that are on the frame and not pixel
      flagged.  It will then find the star with the median FWHM, and
      return the parameters of the fit to that star.
   
      Inputs:
      -------
      dpsf (numpy array, or list)
          The radius to be searched for the psf star.  If this is zero it
          will take the position as fixed.
      fwhm
          The approximate seeing, in pixels.
      noise
          If you have a noise map for the data, specify it with this optional
          keyword.
      plotfile
          Optional output plot of the best-fit psf star.
            
      Outputs.
      -------
      shape_par (3-tuple)
         The three parameters of the PSF:  (fwhm1, fwhm2, angle)
    
      ipsf
         A flag, set to the position in the list of star used, or -1 if no PSF 
         star could be found.
      integer :: nfit
         The number of stars used when deciding which star to fit.'''

      # Setup the plot if requested
      if plotfile is not None:
         fig = plt.figure(constrained_layout=True)
         gs = fig.add_gridspec(2,2)
         ax1 = fig.add_subplot(gs[:,0])
         ax1.set_title('PSF star')
         ax2 = fig.add_subplot(gs[0,1])
         ax2.set_title('Sky fit')
         ax3 = fig.add_subplot(gs[1,1])
         ax3.set_title('PSF fit')

      # Set it up so that if no psf star is found, this is signalled.
      ipsf = -1
      nfit=0
      mfit = 49    ;# maximum number of stars to fit
      a_par = []   ;#  a list of lists of parameters
      idstar = []
      for i in range(len(self.xpsf)):
         self.log('   Trying star {} in the PSF list at {:.2f},{:.2f}'.format(\
            i,self.xpsf[i], self.ypsf[i]))
         # Do an initial check to see if the star is likely to be in frame.
         if (int(self.xpsf[i])<=self.low[0] or int(self.xpsf[i])>=self.high[0] \
          or int(self.ypsf[i])<=self.low[1] or int(self.ypsf[i])>=self.high[1]):
            # The PSF star is too close to the frame edge.
            self.log('      Too close to edge on first check.')
            continue
         
         # The sky box size is determined in the way described in the paper
         # The 453.2 comes from Ns > 100*fwhm**2/(2 log(2)), eqn 16.
         ibox=int(np.sqrt(453.2*fwhm*fwhm+ 4.0*fwhm*fwhm))
         skycnt, skyerr, skynos, cflag = self.skyfit(self.xpsf[i], 
               self.ypsf[i], fwhm, ibox)
         self.log('      @ skyfit gave a sky of {}'.format(skycnt))
         if (cflag != 'O'):
            # The the sky fit failed.  Can't think why this may be, but go to 
            # next psf star anyway.
            self.log('      Skyfit failed with flag {}'.format(cflag))
            continue
         else:
            icurf = 0
         
         # Set up for fitting the PSF star.
         # Set gaussian widths, rotation, normalization and positions.
         if self.profile == 'moffat':
            # initial guess for beta is 2
            w = fwhm/2./np.sqrt(np.sqrt(2) - 1)
         else:
            w = fwhm/1.665
         this_apar = [w,w, 0.0, 1.0, self.xpsf[i], self.ypsf[i], 2]
         if (self.debug):  self.log('      @ Approximate position {},{}'.format(
                          this_apar[4],this_apar[5]))
         # And set normalisation.          
         this_apar[3] = max(self.data[int(self.ypsf[i]),int(self.xpsf[i])]\
               - skycnt, skynos)
         (this_apar, e_pos, cflag, rchi) = self.gfit(0, dpsf, 0, skycnt, 
               skynos, this_apar, noise=noise)
         if (cflag != 'O'):
           self.log('      Fit failed because of flag {}'.format(cflag))
           continue
         # Set the fourth parameter to be the peak signal-to-noise.
         this_apar[3] = this_apar[3]/skynos
         # If you've got to here its a good star.
         idstar.append(i)
         if self.debug: print("this_apr = ",this_apar)
         a_par.append(this_apar)
         nfit=nfit+1
         if self.debug:  self.log("      nfit = {}".format(nfit))
         if (nfit == mfit):  break
   
      a_par = np.array(a_par)
      idstar = np.array(idstar)
      if self.debug: self.log("      nfit = {}".format(nfit))
      if (nfit > 0):
         # Check that there are some stars with peak s:n > 10.
         if (max(a_par[:,3]) < 10.0 or nfit==1):
            # Better take the first good star then.
            i=0
            nfit=1
         else:
            # Throw out all the low signal-to-noise stuff.
            gids = np.nonzero(a_par[:,3] > 10)[0]
            nfit = len(gids)
            a_par = a_par[gids]
            idstar = idstar[gids]
            sids = np.argsort(a_par[:,0]*a_par[:,1])
            i = sids[len(sids)//2]
            self.log("   fwhm1's {}".format(a_par[:,0]))
            self.log("   fwhm2's {}".format(a_par[:,1]))
            self.log("   angles  {}".format(a_par[:,2]))
            self.log("   beta's  {}".format(a_par[:,-1]))
         shape_par = [a_par[i,0], a_par[i,1], a_par[i,2], a_par[i,-1]]
         ipsf=idstar[i]
         if plotfile is not None:
            skycnt, skyerr, skynos, cflag = self.skyfit(a_par[i,4], 
                  a_par[i,5], fwhm, ibox, ax=ax2)

            #this_apar = [w,w, 0.0, 1.0, a_par[i,4], a_par[i,5], 2]
            #this_apar[3] = max(self.data[int(a_par[i,5]),int(a_par[i,4])]\
            #   - skycnt, skynos)
            dummy = self.gfit(1, dpsf, 0, skycnt, skynos, a_par[i], 
                  noise=noise, axes=[ax1,ax3])
            fig.tight_layout()
            fig.savefig(plotfile)
      else:
         shape_par = None; ipsf=-1; nfit=0; rchi=-1
      return (shape_par, ipsf, nfit, rchi)

   def get_fwhm(self, a_par):
      '''returns the fwhm given a_par.  Takes care of whether we're doing a moffat 
       profile (in which case sigma is the shape paramter) or moffat (more 
       complicated)'''
      if self.profile == 'moffat':
         fac = 2*np.sqrt(np.power(2, 1.0/a_par[-1]) - 1)
         fwhm1 = a_par[0]*fac
         fwhm2 = a_par[1]*fac
      else:
         fwhm1 = 1.665*a_par[0]
         fwhm2 = 1.665*a_par[1]
      return(fwhm1,fwhm2)
   
   def extr(self, xpos, ypos, dpos, fwhm, cliprad, shape_par,  
         optnrm, companion, xcomp, ycomp, noise=None, ndither=False, sky=None):
      ''' returns (flux, error, xfit, yfit, xerr, yerr, peak, cflag, skynos_r):
       This subroutine calculates the flux optimally.     
   
       Inputs.
       -------
       xpos, ypos
           The position of the star.
       dpos
           The radius to be searched for the star.  If this is zero it
           will take the position as fixed.
       fwhm
           The approximate seeing, in pixels.
       cliprad
           The radius to clip the mask.
       shape_par
           The three parameters of the PSF.
       optnrm
           The peak flux the photometry is normalised for, divided by the sky
           noise squared.  Zero is sky limit.
       companion
           Is there a companion star?  NOT IMPLEMENTED YET
       xcomp, ycomp
           What's its position?
       noise
           If you have a noise map for the data, specify it using this optinal
           keyword
      
       The returned values
       -------------------
       flux, error
          The flux and its error.
       xfit, yfit
           The fitted position of the star (set to xpos and ypos if position is
           fixed, or fit fails).
       xerr, yerr
           And their error (set to zero if fit fails).
       peak
           The peak flux from the fitted Gaussian, or, if dpos is zero, the
           highest pixel.
       cflag
           A flag returned.  
               O if everything O.K.,  
               E if star too close to frame edge of the frame.
               B if the fit to the sky failed (inherited from skyfit).
               I if the sky is ill-determined (inherited from skyfit).
               M if any of the pixel values within cliprad > datamax
               Or the value of any pixel flags.
       sky,skynos_r
           The sky value, and RMS of the sky.'''
   
      cflag='O'
      # Set the flux and its error to zero, so if the fit fails the values
      # are fairly meaningless.
      flux = 0.0
      error = 0.0
      # The same for the fitted position errors.
      xerr = 0.0
      yerr = 0.0
      # And preserve the position if no fit is made.
      xfit = xpos
      yfit = ypos
   
      # Do an initial check to see if the star is likely to be in frame.
      if (int(xpos)<=self.low[0] + int(fwhm/2) or \
          int(xpos)>=self.high[0]-int(fwhm/2) or \
          int(ypos)<=self.low[1] + int(fwhm/2) or \
          int(ypos)>=self.high[1]-int(fwhm/2) ):
         return(0.0, 0.0, -1, -1, 0.0, 0.0, 0.0, 'E', 0.0, 0.0, 0.0)

      # Measure the sky. The box size is determined in
      # the way described in the paper
      # Measure it first in a smaller box.
      ibox=int(np.sqrt(453.2*fwhm*fwhm+ 4.0*fwhm*fwhm))
      #self.log("Using final ibox: {}".format(ibox))
      skycnt, skyerr, skynos, cflag = self.skyfit(xpos, ypos, fwhm,
          ibox)
      skynos_r=skynos
      sky_flag = cflag
      if (self.debug):  self.log('Returned from skyfit with flag {}'.\
            format(sky_flag))
      if sky_flag != "O":
         return (0.0, 0.0, -1, -1, 0.0, 0.0, 0.0, 'S', 0.0, 0.0, 0.0)
                            
      # Set up the parameters for the fit.
      a_par = np.zeros(7)*0.0
      a_par[0:3] = shape_par[0:3]
      a_par[-1] = shape_par[-1]
      # Set the position.
      a_par[4]=xpos
      a_par[5]=ypos
      if (dpos > 0.0):
         # If the position is free set normalisation to 1, and hunt for 
         # best counts.
         a_par[3]=1.0
         # Set the initial normalisation to the peak counts.
         a_par[3]=max(self.data[int(a_par[5]),int(a_par[4])]-skycnt, skynos)
         # Reset the position.
         a_par[4]=xpos
         a_par[5]=ypos
      else:
         a_par[3]=max(self.data[int(a_par[5]),int(a_par[4])]-skycnt, skynos)
      if (dpos > 0.0 and not ndither):
         (a_par, e_pos, cflag,rchi) = self.gfit(fix_shape=True, dpos=0.0, 
                        fit_sub=False, skycnt=skycnt, skynos=skynos, 
                        a_par=a_par, 
                        noise=noise)
   
         (a_par, e_pos, cflag,rchi) = self.gfit(fix_shape=True, dpos=dpos,
                        fit_sub=False, skycnt=skycnt, skynos=skynos, 
                        a_par=a_par, 
                        noise=noise)
   
         # cflag is deliberately ignored.  We will check for saturation when
         # we do the photometry.
         # Update the positions.
         xfit=a_par[4]
         yfit=a_par[5]
         xerr=e_pos[0]
         yerr=e_pos[1]
      else:
         rchi = 0.0
   
      peak=a_par[3]
   
      # Finally, do the optimal photometry.
      if sky is not None:
         skycnt = sky
      if not ndither or dpos == 0:
         flux, error, cflag = self.sum_flux(skycnt, skyerr, skynos, 
            a_par, cliprad, optnrm, noise=noise)
      else:
         # Dither around the best location and find max flux:
         xpos = a_par[4]
         ypos = a_par[5]
         if not ndither % 2: ndither += 1
         inds = np.indices((ndither,ndither))
         dxs = np.ravel(1.0*(inds[1]-ndither/2)/(ndither/2)*dpos)
         dys = np.ravel(1.0*(inds[0]-ndither/2)/(ndither/2)*dpos)
         flux,error,cflag = self.sum_flux( skycnt, skyerr, skynos,
             a_par, cliprad, optnrm, noise=noise)
         maxi = 0
         for i in range(len(dxs)):
            a_par[4] = xpos + dxs[i]
            a_par[5] = ypos + dys[i]
            this_flux,this_error,this_cflag = self.sum_flux(skycnt, 
                  skyerr, skynos, a_par, cliprad, optnrm, noise=noise)
            if this_flux > flux:
               flux = this_flux  
               error = this_error  
               cflag = this_cflag
               maxi = i
         xfit = xpos + dxs[maxi]
         yfit = ypos + dys[maxi]
   
      if (cflag == 'O'):  cflag=sky_flag
   
      if (self.debug): self.log('Exiting from s/r extr.')
      return (flux, error, xfit, yfit, xerr, yerr, peak, cflag, skycnt, 
            skynos_r, rchi)
   
      
   def skyfit(self, xpos, ypos, seeing, ibox, maxbin=100, ax=None):
      ''' returns (skycnt, skyerr, skynos, cflag):
       This subroutine fits a skewed Gaussian to the pixel distribution
       in an area of sky.
   
        Inputs
        ------
        xpos, ypos
           The first guess position of the star.
        real, intent(in) :: seeing
            The FWHM seeing.
        ibox
           The length of the side of the sky box.
        ax
           Optional matplotlib axes instance to plot the fit
       
        Outputs
        -------
        skycnt, skyerr, skynos
            The estimated sky counts, error, and deviation.
        cflag
            A return flag.  
              B means failed to fit sky.
              I If fit is unreliable because of skewness of sky.
              M if fit is unreliable beacuse of data exceeding datamax'''
      
      # Set default values for the output, to ensure they are defined for
      # compilers which don't set things to zero.  I've chosen 1 for skyerr
      # and skycnt as these are errors, and the code can divide by them
      # later.
      cflag='O'
      skycnt=0.0
      skyerr=1.0
      skynos=1.0
   
      # Form the sky box.  Note that the algorithm used here will not set the
      # box symmetrically about the star if the star is near the array edge.
      # If you want an symmetric box, you can play with the following few 
      # lines of code, and the declaration of jbox.
      x0 = ibox//2
      y0 = ibox//2
      if int(xpos) - ibox//2 < self.low[0]:
         ixslo = self.low[0]
         x0 = int(xpos) - self.low[0]
      else:
         ixslo = int(xpos) - ibox//2
         x0 = ibox//2
      ixslo=max(int(xpos)-ibox//2,self.low[0])
      if int(ypos) - ibox//2 < self.low[1]:
         iyslo = self.low[1]
         y0 = int(ypos) - self.low[1]
      else:
         iyslo = int(ypos) - ibox//2
         y0 = ibox//2
      
      ixshi=min(int(xpos+ibox//2),self.high[0])
      iyshi=min(int(ypos+ibox//2),self.high[1])
      
      # Modal sky estimation.
      # First we find the mean sky.
      clip_array = self.data[iyslo:iyshi,ixslo:ixshi]

      if np.sometrue(clip_array.ravel() > self.datamax):
         cflag = 'M'

      clip_flg = self.pix_flg[iyslo:iyshi,ixslo:ixshi]
      idx = np.indices(clip_array.shape)
      idx[0] = idx[0] - y0
      idx[1] = idx[1] - x0
      mask = np.where((np.power(idx[0],2) + np.power(idx[1],2)) > 4.0*seeing**2,
            1, 0)
      gids = np.nonzero(np.ravel(mask)*np.ravel(clip_flg))
      clip_array = np.ravel(clip_array)[gids]
      icount = len(clip_array)
      if (icount == 0):
        cflag='I'
        return (skycnt, skyerr, skynos, cflag)
   
      #try:
      skymen,skymed,skydev = sigma_clipped_stats(clip_array)
      #except:
      #   cflag = 'B1'
      #   return (skycnt, skyerr, skynos, cflag)
      if (self.debug):
         self.log('@ clip_mean estimates sky is {} +/- {}'.format(skymen,skydev))
         self.log('@ {},{}'.format(xpos, ypos))
   
      # Make a histogram of the sky pixel value distribution from
      # mean-6sigma to mean+6sigma.
      # Actually, let's use Scott's rule for figuring out the bins.
      # Otherwise the histogram can be noisy and the trick below goes
      # terribly wrong.
      ggids = np.greater(clip_array, skymen-6*skydev)*\
              np.less(clip_array, skymen+6*skydev)
      bin_width,bins = scott_bin_width(clip_array[ggids], return_bins=True)
      yhst, xhst = np.histogram(clip_array, bins=bins,
            range=(skymen - 6*skydev, skymen + 6*skydev))
      xhst = (xhst[1:] + xhst[0:-1])/2
      hstep = xhst[1] - xhst[0]
      # Imagine if all the pixels are at values of n+0.25; they will all
      # fall in the bin for value n, which skyfit will fit, giving a sky
      # value that is in error by 0.25 of a pixel.  This cures the problem.
      # First find the modal sky.
      iskymod = np.argmax(yhst)
      if (np.sum(yhst) < 20):
         # Not enough data to fit.
         cflag='B2'
         return (skycnt, skyerr, skynos, cflag)
      # Now total up all the X values in that bin.
      gids = np.nonzero((clip_array > xhst[iskymod])*\
            (clip_array < xhst[iskymod+1]))
      total = np.sum(clip_array[gids])
      xhst = xhst + (total/yhst[iskymod]) - xhst[iskymod]
      # A protection for the fitting routine.  If more than half of the
      # counts have ended up in 1 bin, don't try and fit it (would you
      # believe the answer anyway?).
      if (max(yhst) > 0.5*np.sum(yhst)):
         cflag='B3'
         skyerr=hstep
         skynos=hstep
         skycnt=yhst[np.argmax(xhst)]
         # Don't even bother going around again.
         jflag=0
         sky_par = [0.0,0.0,0.0,0.0]
         sky_chi = 0.0
      else:
         # Otherwise fit it.
         skynos=skydev
         (jflag, skyerr, skynos, sky_chi,sky_par) = skwfit(xhst, yhst, 
               len(bins), skynos)
         skycnt = sky_par[1]
         # Practically, the best the sky fitting routine can do is 0.1 of a 
         # bin.
         skyerr = max(0.1*hstep,skyerr)
      if (jflag != 0):
         # Might want to try a re-bin here...
         cflag='B4'
      
      # Flag it if the chi-squared or the skewness of the sky fit was 
      # too great to be reliable.
      #if (sky_chi    > bad_sky_chi and bad_sky_chi > 0.0 ):  cflag = 'I'
      #if (sky_par[0] > bad_sky_skw and bad_sky_skw > 0.0 ):  cflag = 'I'
      if ax is not None:
         ax.step(xhst, yhst)
         ax.axvline(skycnt, color='red')
         ax.axvline(skycnt-skyerr, color='red', linestyle='--')
         ax.axvline(skycnt+skyerr, color='red', linestyle='--')
         ax.text(0.95, 0.95,"sky count = ${:.2f} \pm {:.2f}$\n"\
                          "sky stddev = {:.2f}".format(skycnt, skyerr, skynos),
                 transform=ax.transAxes, ha='right', va='top', fontsize=10)
      return(skycnt, skyerr, skynos, cflag)


   def sum_flux(self, skycnt, skyerr, skynos, a_par, cliprad, optnrm,
         noise=None):
      ''' returns(flux, error, cflag)
   
       Inputs
       ------
       skycnt, skyerr, skynos:  The estimated sky counts, the error in sky the 
                                determination, and the sky noise.
       a_par:    The profile shape specifiers.  (See function t_gauss for details.)
                 1 and 2 are the FWsHM in orthogonal directions.
                 3 specifies the angle of these directions wrt the pixel grid.
                 4 is the peak flux.  Doesn't matter what is is (within reason),
                   and will normally be set to the value resulting from the fit to
                   the star.
                 5 is the x position of the star.
                 6 is the y position of the star.
       cliprad:  The radius at which the profile is clipped.
       optnrm:   The peak flux of a star for which the extraction is to be 
                 normalised, divided by the RMS noise in the sky (skynos) for that
                 star.  Should be set to zero for sky limited case normalisation.
       noise:    Noise map for the data
   
       The outputs.
       ------------
       error:  The estimated error in sum_flux
       cflag:  An error flag. Set to E if the star is off the edge of the frame, or
               the value of any pixel flags in the weight mask.  If the value 
               is not 'O', then it is the value of the non 'O' flag nearest to 
               the centre of the star. '''
                 
      # The algorithm used is almost identical to that in Naylor (1998).
      # The major change is that the variance array (V in the paper,
      # var in the code here) is calculated as a fraction of the RMS of 
      # the sky (skynos).  This is irrelevant for the weight mask, as 
      # equation 10 of the paper shows it disappears in the normalisation.
   
      cflag='O'
      if abs(a_par[3]) < 1e-10:  
         a_par[3] = 1.0
         cflag = 'Z'
   
      # First calcuate the region of the image to be used.
      min1 = np.floor(int(a_par[4])-cliprad-0.5)
      max1= np.ceil(int(a_par[4])+cliprad+0.5) + 1
      min2 = np.floor(int(a_par[5])-cliprad-0.5)
      max2=np.ceil(int(a_par[5])+cliprad+0.5) + 1
   
      if (min1<self.low[0] or min2<self.low[1] or max1>self.high[0] \
            or max2>self.high[1]):
   
         # The star is too close to the frame edge.  Put in values that
         # won't give divide by zero errors.
         cflag='E'
         sum_flux=1.0
         error=1.0
         if self.debug:
            self._subdata = None
            self._weight = None
         return(sum_flux, error, cflag)
      else:
         if noise is None:
            # Calculate the real variances.
            var_real = skynos*skynos + \
                  (self.data[int(min2):int(max2),int(min1):int(max1)] 
                        -skycnt)/self.gain
            # If the data are very negative, this at least assures an answer.
            np.where(var_real <= 0.0, skynos*skynos, var_real)
         else:
            var_real = np.power(noise[int(min2):int(max2),int(min1):int(max1)], 2)
         
         # Calculate the weight mask.
         norm=0.0
   
         inds = np.indices(var_real.shape)
         ixd = inds[1]*1.0 + min1
         iyd = inds[0]*1.0 + min2
         dist = np.sqrt(np.power(ixd-a_par[4],2) + np.power(iyd-a_par[5],2))
         if self.profile == 'moffat':
            prof = makeMoffat2DEl(a_par)(ixd,iyd)
         else:
            prof = makeGaussian2D(a_par)(ixd,iyd)
         var = skynos*skynos*(1.0 + optnrm*prof/(self.gain*a_par[3]))
         if self.debug:
            self.log("Sky Noise: {}".format(skynos))
         profil = prof/(np.pi*a_par[0]*a_par[1]*a_par[3])
         # linearly interpolate at the clipradius boundary
         profil = np.where((dist < cliprad+0.5)*(dist > cliprad-0.5), 
               profil*(0.5-(dist-cliprad)), profil)
         profil = np.where(dist > cliprad+0.5, 0.0, profil)
         weight = profil/var
         norm = np.sum(np.ravel(profil*profil)/np.ravel(var))
         if self.debug: self.log("Norm:  ".format(norm))
         # This is the weigth mask
         weight = weight/norm
   
      if(self.debug):  
         self.log("sum of weights: {}".format(np.sum(np.ravel(weight))))
         self._subdata = self.data[int(min2):int(max2),int(min1):int(max1)]
         self._weight = weight
      subdata = self.data[int(min2):int(max2),int(min1):int(max1)]
      sum_flux = np.sum(np.ravel(weight)*(np.ravel(subdata)- skycnt))
      # Calculate the error, including that from the sky determination.
      error = np.sqrt( np.sum(np.ravel(var_real*weight*weight)) +\
              skyerr*skyerr*np.sum(np.ravel(weight))**2 )

      # Check to see if pixels exceed datamax
      bids = (dist < cliprad+0.5)*(subdata > self.datamax)
      if np.sometrue(bids):
         cflag = 'M'
   
      return(sum_flux, error, cflag)
   
   
   def gfit(self, fix_shape, dpos, fit_sub, skycnt, skynos, a_par, 
         noise=None, symm=False, axes=None):
      '''  returns (a_par, e_pos, cflag):
       Fits a profile to a star in the array data.  The first guess of the
       parameters is in a_par.  The subroutine returns a_par, the parameters of
       the fit.  You can demand that all the shape parameters are fixed, and that
       only the position and normalisation are free using the logical fix_shape.
       You can demand that the fit is subtracted from the data using the logical
       fit_sub (not implemented in the python version).
   
       You can fit as many profiles as you like, the number being set
       by the size of a_par, but the approximate error in position always
       applies to the first profile.
   
       Inputs (data can be adjusted if the fit is subtracted).
       ------
       fix_shape:  set to 1 for only flux and position variable
       fit_sub:  not implemented
       dpos:     allowable position change
       skycnt:   sky counts
       skynos:   sky noise
       a_par:     initial guess for parameter fit:
                  (amp, x0, y0, std_x, std_y, theta)
       noise:     noise map for the input data
       symm:     force a symmetric PSF
       axes:     optional 2-tuple of MPL axes on which to plot diagnostics
       
       Output.
       -------
       a_par:    shape parameters (see t_gauss for explanation)
       e_pos:    The error in the X and Y position of the first profile.  Set 
                 to zero if the position is fixed.
       cflag:    A flag, normally O, P if fit failed.  or any pixel flag '''
                 
   
      cflag='O'
      
      # Set the position to the center of the sub-array.
      ixcen=int(a_par[4])
      iycen=int(a_par[5])
      
      (fwhm1,fwhm2) = self.get_fwhm(a_par)
   
      # Make a box whose length is twice the fwhm.
      ibox=int(2.0*np.sqrt(fwhm1*fwhm2)) + 1
      if self.debug:  self.log("@ Using ibox = {}".format(ibox))
   
      # Put the data into the 1D array.
      ixbeg=max(self.low[0],ixcen-ibox)
      ixend=min(self.high[0],ixcen+ibox + 1)
      iybeg=max(self.low[1],iycen-ibox)
      iyend=min(self.high[1],iycen+ibox + 1)
   
      subdata = self.data[iybeg:iyend, ixbeg:ixend]
      subflg = self.pix_flg[iybeg:iyend, ixbeg:ixend]

      if noise is not None:
         subnoise = noise[iybeg:iyend, ixbeg:ixend]
      if self.debug:  self.log("skynos = {} skycnt={} gain={}".format(
         skynos, skycnt,self.gain))
      if self.debug:  self.log("max data: {}".format(np.maximum.reduce(np.ravel(subdata))))
      if self.debug:  self.log("min data: {}".format(np.minimum.reduce(np.ravel(subdata))))
      counts = np.where((subdata-skycnt) > 0, (subdata-skycnt), 0)
      if noise is None:
         w = np.where(subflg > 0, 
               np.sqrt(1.0/(skynos*skynos + counts/self.gain)),0.0)
      else:
         w = np.where(subnoise > 0, np.power(subnoise,-1), 0.0)
      icount = len(np.ravel(counts))
      ibad = np.sum(np.ravel(np.where(w > 0.0, 0, 1)))
      
      if (np.sum(np.ravel(w)) < 1e-9):
         if (cflag == 'O'):  cflag='P'
         e_pos=[0.0,0.0]
         return(a_par, e_pos, cflag,0.0)
   
      # Create the models
      if self.profile == 'moffat':
         modi = Moffat2DEl(theta=a_par[2],beta=a_par[-1])
         beta = modi.beta
      else:
         modi = models.Gaussian2D(theta=a_par[2])
         beta = None
   
      if (ibad > int(3.142*a_par[0]*a_par[1]/(1.665*1.665))):
        # A large fraction of the seeing disc is bad pixels.
        # cflag has already been set to what some of the cause was.
        if (cflag == 'O'):  cflag="AB"
        e_pos=[0.0,0.0]
        return(a_par, e_pos, cflag,0.0)
   
      # Normalise the arrays to a maximum value of 1.
      norm_fac=max(np.ravel(counts))
      if (norm_fac < 1e-9): norm_fac=1.0
      counts=counts/norm_fac
      w=w*norm_fac

      # And change any initial guesses of the flux we have (note, this was in 
      # ADU, not counts, so we need to take this into account as well).
      modi.amplitude = a_par[3]/norm_fac
      
      modi.x_mean = a_par[4] - float(ixbeg)
      modi.y_mean = a_par[5] - float(iybeg)
      modi.x_stddev = a_par[0]
      modi.y_stddev = a_par[1]
   
      # Start with the beta-parameter fixed (we'll relax this if this is a 
      # moffat fit):
      if beta is not None: beta.fixed=True
      
      if (fix_shape):
         modi.x_stddev.fixed = True
         modi.y_stddev.fixed = True
      else:
         # Limit the width to be positive, more than 0.5 of a pixel,
         # but not more than 10 times its current value.
         modi.x_stddev.min = 0.5
         modi.x_stddev.max = 10*a_par[0]
         modi.y_stddev.min = 0.5
         modi.y_stddev.max = 10*a_par[1]
         if symm:
            modi.y_stddev.tied = lambda model: model.x_stddev
      # Begin with the rotation fixed.
      modi.theta.fixed = True
      if (dpos > 0.0):
         modi.x_mean.min = modi.x_mean.value-dpos
         modi.x_mean.max = modi.x_mean.value+dpos
         modi.y_mean.min = modi.y_mean.value-dpos
         modi.y_mean.max = modi.x_mean.value+dpos
      else:
         modi.x_mean.fixed = True
         modi.y_mean.fixed = True
      
      if (self.debug):
        self.log('@ ** Now fitting profile with rotation fixed.')
        self.log('@ initial parameters {}'.format(modi.parameters))
   
      sn = counts*w
      if self.debug:  self.log("Maximum s/n: {}".format(max(np.ravel(sn))))
      y,x = np.indices(counts.shape)
      fitter = fitting.LevMarLSQFitter()
      modi = fitter(modi, x, y, counts, weights=w)
      params = modi.parameters
      #errs = np.sqrt(np.diag(fitter.fit_info['cov_x']))
      if (self.debug): self.log("{} {}".format(fitter.fit_info['ierr'], 
                                     fitter.fit_info['message']))
      if (self.debug): self.log('@ Returned from curfit.')
      if (self.debug): self.log('params = {}'.format(params))
      #if (self.debug): self.log('errors = {}'.format(errs))
      icurf = fitter.fit_info['ierr']
      if icurf < 1 or icurf > 4:
         self.log("@  Warning!  rotation-fixed fit failed with {} {}".format(
            icurf, fitter.fit_info['message']))
   
      # Now, let the beta parameter vary, but keep the widths fixed.:
      if (not fix_shape and self.profile == 'moffat'):
         modi.beta.fixed = False
         modi.beta.min = 1.001
         modi.beta.max = 10.0
         modi.x_stddev.fixed = True
         modi.y_stddev.fixed = True
         if (self.debug):
            self.log('@ ** Now fitting profile with beta paramter free.')
         modi = fitter(modi, x, y, counts, weights=w)
         icurf = fitter.fit_info['ierr']
         if icurf < 1 or icurf > 4:
            self.log('@ Warning!  beta-free fit failed with {} {}'.format(
               icurf, fitter.fit_info['message']))
   
      if (not fix_shape and not symm):
         modi.x_stddev.fixed = False
         modi.y_stddev.fixed = False
         if modi.y_stddev.tied is not None: modi.y_stddev.tied = None
         if beta is not None: modi.beta.fixed = True
         # Now let the rotation run free.
         modi.theta.fixed = False
         modi.theta.min = modi.theta.value - 0.78539816
         modi.theta.max = modi.theta.value + 0.78539816
         if (self.debug):
            self.log('@ ** Now fitting profile with rotation free.')
         modi = fitter(modi, x, y, counts, weights=w)
         icurf = fitter.fit_info['ierr']
         if icurf < 1 or icurf > 4:
            self.log('@ Warning!  fit failed with {} {}'.format(icurf, 
               fitter.fit_info['message']))
   
      resids = counts - modi(x, y)
      if self.debug:
         for par in modi.param_names:
            print(par,getattr(modi,par).value)
      # Let's figure out reduced chi-sqr:
      if self.debug:
         self.log("sum^2 resids is {}".format(
                      np.sum(np.power(np.ravel(resids),2))))
         self.log("sub weighted resids is = {}".format(
                      np.sum(np.power(np.ravel(resids)*np.ravel(w),2))))
         self.log("dof is ".format(len(np.ravel(resids)) - len(modi.parameters)))
      rchisq = np.sum(np.power(np.ravel(resids)*np.ravel(w), 2))/\
            (len(np.ravel(resids)) - len(modi.parameters))
      if self.debug:
         self.log("reduced chi-sq is {}".format(rchisq))
   
      # Undo the normalisation
      a_par = [modi.x_stddev.value,modi.y_stddev.value,modi.theta.value,
               modi.amplitude.value*norm_fac*self.gain,
               modi.x_mean.value + float(ixbeg), modi.y_mean.value+float(iybeg)]
      if beta is not None:  a_par.append(modi.beta.value)
   
      if fitter.fit_info['cov_x'] is not None:
         perr = np.sqrt(np.diag(fitter.fit_info['cov_x']))
      else:
         perr = modi.parameters*0 - 1
      if (dpos > 0.0):
        e_pos = [perr[1],perr[2]]
      else:
        e_pos=[0.0,0.0]

      if axes is not None:
         axes[0].imshow(counts, origin='lower', vmin=0, vmax=1)
         el = Ellipse((modi.x_mean.value, modi.y_mean.value),
               width=modi.x_stddev.value, height=modi.y_stddev.value,
               angle=modi.theta.value*360/np.pi, facecolor='none',
               edgecolor='red')
         axes[0].add_patch(el)
         r = np.sqrt((x - modi.x_mean.value)**2 + (y - modi.y_mean.value)**2)
         axes[1].plot(r.ravel(), counts.ravel(), '.')
         sids = np.argsort(r)
         mod = modi(x, y)
         axes[1].plot(r[sids].ravel(), mod[sids].ravel(), '-', color='red', 
               alpha=0.5, zorder=10)
                    
      return(a_par, e_pos, cflag, rchisq)


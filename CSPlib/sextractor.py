'''Source Extractor helper module.'''
from numpy import pi, floor
from astropy.io import fits
from astropy.io import ascii
from .tel_specs import getTelIns
import tempfile
import os
from scipy.stats import exponnorm
import numpy as np

sex_in = '''CATALOG_NAME     {tmpdir}/sextractor.cat
CATALOG_TYPE    ASCII_HEAD
PARAMETERS_NAME {tmpdir}/sextractor.param
DETECT_TYPE     CCD
DETECT_MINAREA  {minarea:d}
DETECT_THRESH   {thresh:.2f}
ANALYSIS_THRESH {thresh:.2f}
FILTER          N
FILTER_NAME     {tmpdir}/sextractor.conv
DEBLEND_NTHRESH 32
DEBLEND_MINCONT {deblend_mc:.3f}
CLEAN           Y
CLEAN_PARAM     1.0
MASK_TYPE       CORRECT
PHOT_APERTURES  {ap_pix:.2f}
PHOT_AUTOPARAMS 2.5, 3.5
SATUR_LEVEL     {datamax:.2f}
MAG_ZEROPOINT   25.000
MAG_GAMMA       4.0 
GAIN            {epadu:.2f}
PIXEL_SCALE     {scale:.5f}
SEEING_FWHM     {fwhm:.3f}
STARNNW_NAME    {tmpdir}/sextractor.nnw
BACK_SIZE       200
BACK_FILTERSIZE 3
CHECKIMAGE_TYPE  NONE
MEMORY_OBJSTACK 2000
MEMORY_PIXSTACK 100000
MEMORY_BUFSIZE  1024
VERBOSE_TYPE    NORMAL
'''

datadir = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data'))

class SexTractor:
   '''A class to runs source extractor on an image.'''

   def __init__(self, image, tel='SWO', ins='NC', scale=None, gain=None):

      self.image = image
      teldata = getTelIns(tel, ins)
      #self.__dict__.update(teldata)
      if isinstance(self.image, str):
         fts = fits.open(image)
      else:
         fts = self.image
      for key in teldata:
         try:
            if isinstance(teldata[key], str):
               if teldata[key][0] == '@':
                  try:
                     self.__dict__[key] = fts[0].header[teldata[key][1:]]
                  except:
                     pass
               else:
                  self.__dict__[key] = teldata[key]
            else:
               self.__dict__[key] = teldata[key]
         except:
            pass

      if scale is not None: self.scale = scale
      if gain is not None: self.gain = gain
      self.tmpdir = tempfile.mkdtemp(dir='.')
      self.tab = None

   def makeSexFiles(self, aper, datamax, fwhm, thresh, deblend_mc=0.005):
      '''Output a sextractor config file.

      Args: 
         aper(float): aperture in arc-sec
         datamax(float): saturation limit
         fwhm (float): FWHM of stars for this night
         thresh (float): threshold for detections in units of sigma.

      Returns:
         None

      Effects:
         Creates two files:  sextractor.in and sextractor.param
         Copies sextractor.nnm to local fodler.'''

      scale = self.scale
      ap_pix = aper/scale
      minarea = int(floor(pi*(fwhm/scale)**2/4))
      if minarea == 0: minarea = 3
      epadu = self.gain
      tmpdir = self.tmpdir

      with open(os.path.join(tmpdir,'sextractor.in'),'w') as fout:
         fout.write(sex_in.format(**locals()))
 
      with open(os.path.join(tmpdir,'sextractor.param'), 'w') as fout:
         fout.write('MAG_APER(1)\nX_IMAGE\nY_IMAGE\nFWHM_IMAGE\nCLASS_STAR\n')
         fout.write('FLAGS\n')

      os.system('cp {} {}'.format(os.path.join(datadir, 'sextractor.nnw'),
         tmpdir))

   def cleanup(self):
      for fil in ['sextractor.in','sextractor.cat','sextractor.nnw',
            'sextractor.param','image.fits']:
         fname = os.path.join(self.tmpdir, fil)
         if os.path.isfile(fname): os.unlink(fname)
      os.rmdir(self.tmpdir)

   def run(self, fwmin=0.7, fwmax=2.5, thresh=3, datamax=30000,
         Nmax=None, deblend_mc=0.005):

      aper = fwmin+fwmax
      fwhm = fwmin+(fwmin+fwmax)/4
      self.makeSexFiles(aper, datamax, fwhm, thresh, deblend_mc)
      if not isinstance(self.image, str):
         image = os.path.join(self.tmpdir, 'image.fits')
         self.image.writeto(image, overwrite=True)
      else:
         image = self.image

      ret = os.system('sex {} -c {}/sextractor.in'.format(image,
         self.tmpdir))
      if ret != 0:
         raise RuntimeError('Sextractor failed')

   def parseCatFile(self):
      '''Read in the catalog data.'''
      if not os.path.isfile(os.path.join(self.tmpdir,'sextractor.cat')):
         raise IOError('Sextractor catalog file not found, did you run()?')
      self.tab = ascii.read(os.path.join(self.tmpdir,'sextractor.cat'))
      return self.tab

   def filterStars(self, fmin=0, fmax=20, scale=1, nsigma=5):
      if self.tab is None:
         raise RuntimeError("You have to run() and parseCatFile first")

      gids = np.greater(self.tab['FWHM_IMAGE'], fmin*scale)*\
             np.less(self.tab['FWHM_IMAGE'], fmax*scale)
      lamb,mu,sig = exponnorm.fit(self.tab['FWHM_IMAGE'][gids])

      gids = gids*np.greater(self.tab['FWHM_IMAGE'], mu-nsigma*sig)*\
             np.less(self.tab['FWHM_IMAGE'], mu + nsigma*sig)
      return(self.tab[gids])






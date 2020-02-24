'''Source Extractor helper module.'''
from numpy import pi, floor
from astropy.io import fits
from astropy.io import ascii
from .tel_specs import getTelIns
import os

sex_in = '''CATALOG_NAME     sextractor.cat
CATALOG_TYPE    ASCII_HEAD
PARAMETERS_NAME sextractor.param
DETECT_TYPE     CCD
DETECT_MINAREA  {minarea:d}
DETECT_THRESH   {thresh:.2f}
ANALYSIS_THRESH {thresh:.2f}
FILTER          N
FILTER_NAME     sextractor.conv
DEBLEND_NTHRESH 32
DEBLEND_MINCONT 0.005
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
STARNNW_NAME    sextractor.nnw
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

   def __init__(self, image, tel='SWO', ins='NC'):

      self.image = image
      teldata = getTelIns(tel, ins)
      self.__dict__.update(teldata)

   def makeSexFiles(self, aper, datamax, fwhm, thresh):
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

      with open('sextractor.in','w') as fout:
         fout.write(sex_in.format(**locals()))
 
      with open('sextractor.param', 'w') as fout:
         fout.write('MAG_APER(2)\nX_IMAGE\nY_IMAGE\nFWHM_IMAGE\nCLASS_STAR\n')
         fout.write('FLAGS\n')

      os.system('cp {} .'.format(os.path.join(datadir, 'sextractor.nnw')))

   def cleanup(self):
      for fil in ['sextractor.in','sextractor.cat','sextractor.nnw',
            'sextractor.param']:
         os.unlink(fil)

   def run(self, fwmin=0.7, fwmax=2.5, thresh=3, datamax=30000,
         Nmax=None):

      aper = fwmin+fwmax
      fwhm = fwmin+(fwmin+fwmax)/4
      self.makeSexFiles(aper, datamax, fwhm, thresh)
      ret = os.system('sex {} -c sextractor.in'.format(self.image))
      if ret != 0:
         raise RuntimeError('Sextractor failed')

   def parseCatFile(self):
      '''Read in the catalog data.'''
      if not os.path.isfile('sextractor.cat'):
         raise IOError('Sextractor catalog file not found, did you run()?')
      tab = ascii.read('sextractor.cat')
      return tab





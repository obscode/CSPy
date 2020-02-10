'''CCD reduction routines.'''

import os
from os.path import realpath,join,dirname,isfile,isdir
import numpy as np
from astropy.io import fits
from .tel_specs import getTelIns
import re

slice_pat = re.compile(r'\[([0-9]+):([0-9]+),([0-9]+):([0-9]+)\]')

# Where we keep all the calibration frames
if 'DATADIR' not in os.environ:
   # assume it's with the module
   datadir = join(realpath(dirname(__file__)), 'data')
else:
   datadir = os.environ['DATADIR']
if not isdir(datadir):
   raise RuntimeError("Data directory {} not found".format(datadir))

def getBackupCalibration(typ='Zero', chip=1, filt=None, tel='SWO',
      ins='NC'):
   if typ not in ['Zero','Flat','Shutter']:
      raise ValueError("Type must be Zero, Flat, Shutter")
   if typ == 'Zero':
      filename = join(datadir, tel+ins, "Zeroc{}.fits.bz2".format(chip))
   elif typ == 'Flat':
      filename = join(datadir, tel+ins, "SFlat{}c{}.fits.bz2".format(filt,chip))
   else:
      filename = join(datadir, tel+ins, "SH{}.fits.bz2".format(chip))

   fts = fits.open(filename)
   return fts

def bias_correct(fts, overscan=True, frame=None, outfile=None, tel='SWO',
      ins='NC'):
   '''Apply a bias correction from the overscan and bias frame if supplied.

   Args:
      fts (file or FITS): Frame to correct
      overscan(bool): compute bias from overscan section of CCD
      frame (file or FITS):  If not None, a BIAS frame to subtract
      outfile(str): If not None, bias-corrected frame is saved as filename

   Returns:
      New fits object with bias corrected data, trimmed to DATASEC, if set'''

   if 'DATASEC' in fts[0].header:
      x0,x1,y0,y1 = slice_pat.search(fts[0].header['DATASEC']).groups()
      x0 = int(x0)-1
      x1 = int(x1)-1
      y0 = int(y0)-1
      y1 = int(y1)-1
   else:
      x0 = 0; x1 = fts[0].header['NAXIS1']-1
      y0 = 0; y1 = fts[0].header['NAXIS2']-1
   xslc = slice(x0,x1+1)
   yslc = slice(y0,y1+1)
   # Trim
   newdata = fts[0].data[yslc,xslc]*1.0   # ensure float type
   newdata = newdata.astype(np.float32)

   # Compute BIAS from CCD
   specs = getTelIns(tel,ins)
   newhdr = fts[0].header.copy()

   # Overscan on axis 0 (NAXIS1)
   if 1 in specs['overscan']:
      lov,hov = specs['overscan'][1]
      bdata = np.mean(fts[0].data[yslc,lov:hov], axis=1)
      newdata = newdata - bdata[:,np.newaxis]
      newhdr['COMMENT'] = "BIAS corrected using ax=1. Avg={:.1f}".format(
            np.mean(bdata))

   # Overscan on axis 1 (NAXIS2)
   if 0 in specs['overscan']:
      lov,hov = specs['overscan'][0]
      bdata = np.mean(fts[0].data[lov,hov:xslc], axis=0)
      newdata = newdata - bdata[np.newaxis,:]
      newhdr['COMMENT'] = "BIAS corrected using ax=0. Avg={:.1f}".format(
            np.mean(bdata))

   if frame is not None:
      if type(frame) is str:
         biasfts = fits.open(frame)
      else:
         biasfts = frame
      if biasfts[0].data.shape != newdata.shape:
         raise ValueError("BIAS frame has incorrect shape")

      newdata = newdata - biasfts[0].data
      newhdr['COMMENT'] = "BIAS corrected using frame. Avg={:.1f}".format(
            np.mean(biasfts[0].data))

   newhdu = fits.PrimaryHDU(data=newdata, header=newhdr)
   newhdu.scale('float32')
   newfts = fits.HDUList([newhdu])
   if outfile is not None:
      newfts.writeto(outfile, overwrite=True)
   return newfts


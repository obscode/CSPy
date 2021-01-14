'''CCD reduction routines.'''

import os
from os.path import realpath,join,dirname,isfile,isdir
import numpy as np
from scipy.stats import mode
from astropy.io import fits
from .tel_specs import getTelIns
from . import fitsutils
import re
from .irafstuff import imcombine

slice_pat = re.compile(r'\[([0-9]+):([0-9]+),([0-9]+):([0-9]+)\]')

# Cache variables (we we don't open/close too much)
shutters = {}

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
   '''Utility function for retrieving calibration data from the correct
   location.

   Args:
     typ (str):  The type of calibration: 'Zero','Flat', or 'Shutter'
     chip (int):  Chip number
     filt (str):  Which filter (for calibrations that are filter specific)
     tel (str):  Telescope code (SWO, DUP, etc)
     ins (str):  Instrument code (DC, NC, RC, etc)

   Returns:
      calibration (fits instance):  the calibration as a fits object.
   '''

   if typ not in ['Zero','Flat','Shutter']:
      raise ValueError("Type must be Zero, Flat, Shutter")
   if typ == 'Zero':
      filename = join(datadir,tel+ins,"CAL", "Zeroc{}.fits".format(chip))
   elif typ == 'Flat':
      filename = join(datadir,tel+ins,"CAL", "SFlat{}c{}.fits".format(filt,chip))
   else:
      filename = join(datadir,tel+ins,"SH{}.fits".format(chip))

   found = False
   for ext in ['','.gzip','.bzip2']:
      print(filename+ext)
      if os.path.isfile(filename+ext):
         fts = fits.open(filename+ext)
         found = True
         break
   if not found:
      raise IOError("Error, no calibration file found. Check DATADIR")

   return fts

def makeBiasFrame(blist, outfile=None, tel='SWO', ins='NC'):
   '''Given a set of BIAS frames, combine into a single frame using
   imcombine.
   
   Args:
      blist (list): List of bias frames to combine
      outfile (str):  Output FITS file for combined BIAS
      tel(str):  Telescope code (e.g., SWO)
      ins(str):  Instrument code (e.g. DC)
   Returns:
      BIAS (fits instance):  combined BIAS frame as fits object
   '''
   specs = getTelIns(tel,ins)

   res = imcombine(blist, combine='average', reject='avsigclip', 
         lsigma=3, hsigma=3, nkeep=1)
   xl,xh,yl,yh = specs['datasec']
   A = res[0].data[yl-1:yh,xl-1:xh]     # index from zero, not one
   if 1 in specs['overscan']:
      lov,hov = specs['overscan'][1]
      O = res[0].data[yl:yh+1, lov:hov+1]
      OV = np.mean(O, axis=1)
      OV2 = np.concatenate([[OV[0]],0.5*OV[1:-1]+0.25*OV[:-2]+0.25*OV[2:],
         [OV[-1]]])
      A = A - OV2[:,np.newaxis]
   elif 2 in specs['overscan']:
      lov,hov = specs['overscan'][2]
      O = res[0].data[lov:hov+1, xl:xh+1]
      OV = np.mean(O, axis=0)
      OV2 = np.concatenate([[OV[0]],0.5*OV[1:-1]+0.25*OV[:-2]+0.25*OV[2:],
         [OV[-1]]])
      A = A - OV2[np.newaxis,:]

   phdu = fits.PrimaryHDU(A.astype(np.float32), header=res[0].header)
   fts = fits.HDUList([phdu])
   if outfile is not None:
      fts.writeto(outfile, overwrite=True)
   return fts


def biasCorrect(image, overscan=True, frame=None, outfile=None, tel='SWO',
      ins='NC', verbose=False):
   '''Apply a bias correction from the overscan and bias frame if supplied.

   Args:
      image (file or FITS): Frame to correct
      overscan(bool): compute bias from overscan section of CCD
      frame (file or FITS):  If not None, a BIAS frame to subtract
      outfile(str): If not None, bias-corrected frame is saved as filename

   Returns:
      New fits object with bias corrected data, trimmed to DATASEC, if set'''

   if isinstance(image, str):
      fts = fits.open(image)
   else:
      fts = image
   if 'DATASEC' in fts[0].header:
      x0,x1,y0,y1 = slice_pat.search(fts[0].header['DATASEC']).groups()
      x0 = int(x0)-1
      x1 = int(x1)-1
      y0 = int(y0)-1
      y1 = int(y1)-1
      if verbose:
         print("Found DATASEC, trimming to [{}:{},{}:{}]".format(x0,x1,y0,y1))
   else:
      x0 = 0; x1 = fts[0].header['NAXIS1']-1
      y0 = 0; y1 = fts[0].header['NAXIS2']-1
      if verbose:
         print("No DATASEC, keeping original dimensions")

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
      if verbose:
         print("BIAS section: [{}:{},*]".format(lov,hov))
      bdata = np.mean(fts[0].data[yslc,lov:hov], axis=1)
      newdata = newdata - bdata[:,np.newaxis]
      comm = "BIAS corrected using ax=1. Avg={:.1f}".format(np.mean(bdata))
      newhdr['COMMENT'] = comm
      if verbose:
         print(comm)

   # Overscan on axis 1 (NAXIS2)
   if 0 in specs['overscan']:
      lov,hov = specs['overscan'][0]
      if verbose:
         print("BIAS section: [*,{}:{}]".format(lov,hov))
      bdata = np.mean(fts[0].data[lov,hov:xslc], axis=0)
      newdata = newdata - bdata[np.newaxis,:]
      comm = "BIAS corrected using ax=0. Avg={:.1f}".format(np.mean(bdata))
      newhdr['COMMENT'] = comm
      if verbose:
         print(comm)

   if frame is not None:
      if type(frame) is str:
         biasfts = fits.open(frame)
      else:
         biasfts = frame
      if biasfts[0].data.shape != newdata.shape:
         raise ValueError("BIAS frame has incorrect shape")

      newdata = newdata - biasfts[0].data
      comm = "BIAS corrected using frame. Avg={:.1f}".format(
            np.mean(biasfts[0].data))
      newhdr['COMMENT'] = comm
      if verbose:
         print(comm)

   newhdu = fits.PrimaryHDU(data=newdata, header=newhdr)
   newhdu.scale('float32')
   newfts = fits.HDUList([newhdu])
   if outfile is not None:
      newfts.writeto(outfile, overwrite=True)
   return newfts

def LinearityCorrect(fts, copy=False, tel='SWO',ins='NC', chip='@OPAMP'):
   '''Perform linearity correction on the data. 

   Args:
      fts (fits object):  Data frame to correct
      copy (bool):  If true, copy the data before modifying and return copy
                    Otherwise, modify data in-place and return
      tel (str):  Telescope code
      ins (str):  Instrument code
      chop (str, int, or None):  If None, there is only one chip, so not needed,
                  if int, use that as chip number, if string starting with @,
                  use as header keyword

   Returns:
      fts object.  The original (with comment added) or new object'''

   if copy:
      fts = fitsutils.copyFits(fts)

   lincors = getTelIns(tel,ins)['lincorr']
   if chip is not None:
      if type(chip) is str and chip[0] == '@':
         chip = fts[0].header[chip[1:]]
      if chip not in lincors:
         raise ValueError('chip {} not found in tel_specs for {}{}'.format(
            chip,tel,ins))
      lincors = lincors[chip]
   alpha = lincors['alpha']
   c1 = lincors['c1']
   c2 = lincors['c2']
   c3 = lincors['c3']

   spix = fts[0].data/32000.0
   fcor = alpha*(1.0 + c2*spix + c3*np.power(spix,2))
   fts[0].data = fts[0].data*fcor

   fts[0].header['COMMENT'] = "Linearity correction applied with:"
   fts[0].header['COMMENT'] = \
         "   c1={:.5f},c2={:.5f},c3={:.5f},alpha={:.5f}".format(c1,c2,c3,alpha)
   return fts

def ShutterCorrect(fts, frame=None, copy=False, tel='SWO',ins='NC', 
      chip='@OPAMP', exptime='@EXPTIME'):
   '''Perform linearity correction on the data. 

   Args:
      fts (fits object):  Data frame to correct
      copy (bool):  If true, copy the data before modifying and return copy
                    Otherwise, modify data in-place and return
      tel (str):  Telescope code
      ins (str):  Instrument code

   Returns:
      fts object.  The original (with comment added) or new object'''
   global shutters

   if copy:
      fts = fitsutils.copyFits(fts)

   if chip is not None:
      if type(chip) is str and chip[0] == '@':
         chip = fts[0].header[chip[1:]]

   if type(exptime) is str and exptime[0] == '@':
      exptime = fts[0].header[exptime[1:]]

   if frame is None:
      if (tel,ins,chip) not in shutters:
         fname = join(datadir, tel+ins, "SH{}.fits".format(chip))
         shutters[(tel,ins,chip)] = fits.open(fname)
      frame = shutters[(tel,ins,chip)]
   else:
      if isinstance(frame, str):
         frame = fts.open(frame)
   factor = frame[0].data/exptime
      
   fts[0].data = fts[0].data / (1.0 + factor)
   fts[0].header['COMMENT'] = "Shutter correction using EXPT={}".format(exptime)

   return fts


def makeFlatFrame(flist, outfile=None, tel='SWO', ins='NC'):
   '''Given a set of sky flat frames, combine into a single frame using
   imcombine. It is assumed the flats have already been corrected for
   BIAS, dark, etc.
   
   Args:
      flist(str,list):  Input images to make the flat. They can be specified
                        as a list of filenames, list of FITS objecst, a
                        glob patter, or file with list beginning with '@'
      outfile(str):  If specified, output the flat to this file
      tel:  Telescope code (e.g., SWO)
      ins:  Instrument code (e.g. NC)
      
   Returns:
      FITS instance containing the flat
   '''
   specs = getTelIns(tel,ins)

   res = imcombine(flist, combine='median', reject='sigclip', 
         lsigma=3, hsigma=3, nkeep=1)

   # Determine the mode of the pixels
   if 'statssec' in specs:
      x0,x1,y0,y1 = specs['statssec']
      y0 = y0 - 1   # FITS index from 1, not zero
      x0 = x0 - 1
      subdata = res[0].data[y0:y1,x0:x1]
   else:
      subdata = res[0].data

   mod,cnt = mode(subdata.ravel()) 
   mod = mod[0]
   
   # Re-scale flat by the mode
   res[0].data = res[0].data/mod

   res.writeto(outfile, overwrite=True)
   return res

def flatCorrect(image, flat, outfile=None, replace=1.0):
   '''Flat field correct list of images.

   Args:
      image (str or FITS):  filename of FITS instance to correct
      flat (fits, or file):  Flat field. Either a FITS instance, or filename
      outfile (str):  If specified, output the corrected image to outfile
      replace (float):       In case of division by zero, replacement value

   Returns:
      FITS instance of the corrected field.
   '''
   
   if isinstance(image, str):
      image = fits.open(image)
   newhdr = image[0].header.copy()

   if isinstance(flat, str):
      flatfts = fits.open(flat)
      newhdr['COMMENT'] = "Flat field corrected using {}".format(flat)
   else:
      flatfts = flat

   corr = np.where(np.equal(flat[0].data, 0.0), replace, 
         image[0].data/flat[0].data)

   newhdu = fits.PrimaryHDU(data=corr, header=newhdr)
   newhdu.scale('float32')
   newfts = fits.HDUList([newhdu])
   if outfile is not None:
      newfts.writeto(outfile, overwrite=True)
   return newfts

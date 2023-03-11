'''Some FITS-related routines that are used again and again. These are for
simple CCD frames (one PrimaryHDU in and HDUList)'''

from astropy.io import fits
import warnings

def qdump(filename, data, header=None, extras=None):
   '''quickly produce a FITS file with data and header from another FITS fits.
   Args:
      filename(str):  Name of output file
      data(float arr): data to dump
      header(str or header): Header to use or file with fits header
      extras(dict):  Extra key-word/value to add to FITS file
   Returns:
      None.
   Effetcs:
      fits file is created.
   '''

   if header is None:
      header = fits.PrimaryHDU(data)
      if extras is not None:
         for key in extras:
            header[0].header[key] = extras[key]
      header.writeto(filename, overwrite=True)
   else:
      if type(header) is str:
         hfts = fits.open(header)
         header = hfts[0].header
      if extras is not None:
         for key in extras:
            header[key] = extras[key]
      fits.writeto(filename, data, header, overwrite=True)

def copyFits(inp):
   '''Copy the FITS header and data and return the copy.'''
   newhdr = inp[0].header.copy()
   newdata = inp[0].data.copy()
   newphdu = fits.PrimaryHDU(newdata, header=newhdr)
   newhdul = fits.HDUList([newphdu])
   return newhdul



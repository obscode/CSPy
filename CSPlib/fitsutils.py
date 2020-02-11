'''Some FITS-related routines that are used again and again. These are for
simple CCD frames (one PrimaryHDU in and HDUList)'''

from astropy.io import fits

def copyFits(inp):
   '''Copy the FITS header and data and return the copy.'''
   newhdr = inp[0].header.copy()
   newdata = inp[0].data.copy()
   newphdu = fits.PrimaryHDU(newdata, header=newhdr)
   newhdul = fits.HDUList([newphdu])
   return newhdul



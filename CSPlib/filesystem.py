'''Various routines for dealing with files on the filesystem.'''
from astropy.io import fits
from glob import glob
from astropy.time import Time

def CSPname(fitsfile, idx=1, suffix='.fits'):
   '''Given a file or fits instance, get a filename using the CSP convention.
   If the file already exists, index it accordingly.

   Args:
      fitsfile (str or fits instance):  the input FITS file or astropy.io.fits
                                        instance.
      idx (int): A running index (for repeated observations) default:1
      suffix (str): The suffix for the file. Default:  .fits

   Returns:
      filename (str):  The filename using CSP convention.'''


   template = "{obj}_{filt}{idx:02d}_{tel}_{ins}_{YY}_{MM:02d}_{DD:02d}{suf}"
   
   if isinstance(fitsfile, fits.HDUList):
      fts = fitsfile
   elif isinstance(fitsfile, str):
      fts = fits.open(fitsfile)
   else:
      raise ValueError("fitsfile must be a filename or astropy.io.fits "\
            "instance")

   # Now, DATE-OBS changes at midnight. So the better way to get the "day"
   # is to add 0.5 to JD (push it beyond change-over and get the year/month/day
   jd = Time(fts[0].header['JD'] + 0.5, format='jd')
   dt = jd.to_datetime()
   YY,MM,DD = dt.year,dt.month,dt.day
   YY = int(YY)
   MM = int(MM)
   DD = int(DD)
   
   args = dict(
       obj=fts[0].header['OBJECT'],
       filt=fts[0].header['FILTER'],
       YY=YY, MM=MM, DD=DD,
       tel=fts[0].header['TELESCOP'],
       ins=fts[0].header['INSTRUM'],
       suf=suffix, idx=idx)
   
   name = template.format(**args)
   return name

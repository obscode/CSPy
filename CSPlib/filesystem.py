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

def utname(fitsfile, suffix='.fits'):
   '''Given a file or fits instance, get a filename using the LCO convention.

   Args:
      fitsfile (str or fits instance):  the input FITS file or astropy.io.fits
                                        instance.
      suffix (str): The suffix for the file. Default:  .fits

   Returns:
      filename (str):  The filename using LCO convention.'''


   template = "ut{YY}{MM:02d}{DD1:02d}_{DD2:02d}/{fname}{suf}"
   
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
   # Only take last two digits of YY
   YY = int(YY) % 1000 
   MM = int(MM)
   DD2 = int(DD)
   DD1 = DD2-1
   
   args = dict(
       YY=YY, MM=MM, DD1=DD1, DD2=DD2,
       fname=fts[0].header['FILENAME'],
       suf=suffix)
   
   name = template.format(**args)
   return name

def LCOGTname(fitsfile, idx=1, suffix='.fits'):
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
   jd = Time(fts[1].header['MJD-OBS'] + 0.5, format='mjd')
   dt = jd.to_datetime()
   YY,MM,DD = dt.year,dt.month,dt.day
   YY = int(YY)
   MM = int(MM)
   DD = int(DD)
   
   args = dict(
       obj=fts[1].header['OBJECT'],
       filt=fts[1].header['FILTER'],
       YY=YY, MM=MM, DD=DD,
       tel=fts[1].header['ORIGIN'],
       ins=fts[1].header['INSTRUME'],
       suf=suffix, idx=idx)
   
   name = template.format(**args)
   return name

'''A module for updating FITS headers, for use by CSP pipeline.'''
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from math import floor
from CSPlib import wairmass_for_lco_images

obstypes = {'dflat':'dflat',
            'sflat':'sflat',
            'flat':'sflat',
            'object':'astro',
            'astro':'astro',
            'bias':'zero',
            'zero':'zero',
            'none':'none',
            'test':'none',
            'linear':'none',
            'shutter':'none',
            'focus':'none',
            'pointing':'none',
            }

# A map from WHEEL1,WHEEL2 values to simpler strings
filter_map = {
   "B (LC3013)":"B",
   "V9844":"V",
}

filter_combos = {
   ('ND 0.9','Halpha-off'):'H',
   ('ND 0.9','7415'):'C',
}

def shift_center(header):
   '''Shift the RA/DEC from center of detector to center of chip.
   
   Args:
      header (astropy.fits header):  Header of the image.

   Returns:
      (RA,DEC):  tuple of new center coordinates.
    '''

   chip = int(header.get('OPAMP',0))
   jd = header['JD']
   RA = header['RA']
   DEC = header['DEC']
   coord = SkyCoord(RA, DEC, unit=(u.hourangle, u.degree), frame='icrs')

   if chip == 0:
      # No OPAMP, the FITS file was probably stiched
      newra = coord.ra.to_string(unit=u.hourangle, sep=":", precision=2,
                                 pad=True)
      newdec = coord.dec.to_string(unit=u.degree, sep=":", precision=2,
                                 pad=True, alwayssign=True)
      return(newra,newdec)

   offset = 10.5*u.arcmin
   if ( jd < 2456871.917):
      pa = [-45, 45, 135, -135][chip-1]
   else:
      pa = [45, 135, -135, 45][chip-1]
      
   newcoord = coord.directional_offset_by(pa*u.degree, offset)
   newra = newcoord.ra.to_string(unit=u.hourangle, sep=":", precision=2,
         pad=True)
   newdec = newcoord.dec.to_string(unit=u.degree, sep=":", precision=2,
         pad=True, alwayssign=True)
   return(newra,newdec)

def update_header(f, fout=None):
   '''Do various housekeeping on an input header.

   Args:
       f (str):  filename
       fout (str):  output file name if not None

   Returns:
       astropy.io.fits instance of updated FITS
   
   Effects:
      Output file created
   '''
   fts = fits.open(f, memmap=False)
   h = fts[0].header

   # Get rid of spaces!!  First, leading/trailing spaces
   if h['OBJECT'][0] == ' ' or h['OBJECT'][-1] == ' ':
      h['OBJECT'] = h['OBJECT'].strip()
   # Now spaces in the middle
   if h['OBJECT'].find(' ') > 0:
      h['OBJECT'] = h['OBJECT'].replace(' ','_')

   # Deal with filters.
   if h['FILTER'] == 'COMBO':
      # Combination of two filters. Check map, otherwise concatenate
      if (h['WHEEL1'],h['WHEEL2']) in filter_combos:
         h['FILTER'] = filter_combos[(h['WHEEL1'],h['WHEEL2'])]
      else:
         w1 = h['WHEEL1'].replace(' ','')
         w2 = h['WHEEL2'].replace(' ','')
         h['FILTER'] = w1+'+'+w2
   elif len(h['FILTER']) > 0:
      h['FILTER'] = h['FILTER'][0]

   # First, update the OBSTYPE, as per PREV_SWONC
   obj = h['OBJECT'].lower()
   exptype = h['EXPTYPE'].lower()
   if exptype == 'focus':
      obstype = 'focus'
   elif exptype == 'bias':
      obstype = 'bias'
   elif exptype == 'flat':
      obstype = 'sflat'
   else:
      # Get rid of spaces!!!
      obj = obj.replace(' ','_')
      if obj not in obstypes:
         if obj[-4:] == '_bad':
            obstype = 'none'
         elif obj.find('focus') >= 0:
            obstype = 'focus'
         else:
            obstype = 'astro'
      else:
         obstype = obstypes[obj]
   # udpate:
   h['OBSTYPE'] = (obstype, "Type of image.")
   
   # Next, set keywords as per HEADERS
   ra = h['RA']
   dec = h['DEC']
   eq = h['EQUINOX']
   etime = h['EXPTIME'] 
   obj = h['OBJECT']
   utstart = h['UT-TIME']
   date = h['DATE-OBS'].split('T')[0]
   yyyy,mm,dd = date.split('-')
   date = "{:04d}-{:02d}-{:02d}".format(int(yyyy),int(mm),int(dd))
   h['DATE-OBS'] = (date, "UT DATE AT START OF FRAME")

   wair,stmid,utmid = wairmass_for_lco_images(ra, dec, eq, date,
         utstart, etime)

   tsidh = int(floor(stmid.hour))
   tsidm = int(floor((stmid.hour-tsidh)*60))
   tsids = (stmid.hour-tsidh-tsidm/60)*3600

   h['TELESCOP'] = ("SWO", "Telescope. CSP Keyword")
   h['INSTRUM'] = ("NC", "Instrument. CSP Keyword")
   h['JD'] = (utmid.jd, 'JD at mid exposure. CSP keyword')
   h['NEWRA'] = (h['RA'], "Right Ascension. CSP Keyword")
   h['NEWDEC'] = (h['DEC'], "Declination. CSP Keyword")
   h['NEWEXPT'] = (etime, "Exposure time. CSP Keyword")
   h['NEWST'] = ("{:d}:{:d}:{:.2f}".format(tsidh,tsidm,tsids),
                 "Sidereal time at mid exposure. CSP keywrod")
   h['UTMID'] = ("{:d}:{:d}:{:.1f}".format(utmid.datetime.hour,
          utmid.datetime.minute, utmid.datetime.second), 
          "UT time at mid exposure. CSP Keyword")
   h['WAIRMASS'] = (round(wair,4), "Weighted (effective) airmass. CSP Keyword")

   # Shift RA/DEC to the center of this chip
   newra,newdec = shift_center(h)
   h['RA'] = newra
   h['DEC'] = newdec
   if fout is not None:
      fts.writeto(fout, overwrite=True, output_verify='fix')
   return fts


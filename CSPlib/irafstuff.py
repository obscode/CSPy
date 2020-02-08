'''module for IRAF replacements.'''
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from numpy import pi,sqrt,sin

lco = EarthLocation.of_site('lco')

def wairmass_for_lco_images(ra, dec, equinox, dateobs, utstart, exptime,
      scale=750.):
   '''Compute the effective airmass for observations at LCO. 

   Args:
      - ra(str):       Right-ascention in the format hh:mm:ss.s
      - dec(str):      Declination in the format +dd:mm:ss.s
      - equinox(float): Equinox in decimal years
      - dateobs(str): Date of observation in the format yyyy-mm-dd
      - utstart(str):    Start UT time in the format hh:mm:ss
      - exptime(float): Exposure time in seconds
      - scale (float): Scale hight of atmosphere

   Returns:
      (AM, ST, UTmid):  Airmass (float), 
                        sidereal time (astropy.Time)
                        UT time at mid-exposure (astropy.Time)'''

   eq = Time(equinox, format='decimalyear')
   obj = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), equinox=eq)
   obstime = Time("{}T{}".format(dateobs, utstart), format='isot', scale='utc',
         location=lco)
   utmid = obstime + exptime*u.s/2
   tsid = utmid.sidereal_time('apparent')
   utend = obstime + exptime*u.s

   altaz = obj.transform_to(AltAz(obstime=[obstime, utmid, utend],
      location=lco))
   elev = altaz.alt.value*pi/180
   x = scale*sin(elev)
   AMs = sqrt(x**2 + 2*scale + 1) - x
   AM = (AMs[0] + 4*AMs[1] + AMs[2])/6
   return(AM, tsid, utmid)

   



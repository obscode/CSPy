'''Module for dealing with catalog files of local sequende stars.'''

from astropy.io import ascii,fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u
from CSPlib import sextractor,getPS
from numpy import sqrt, newaxis, sum, greater, less, arange, argsort

def make_ls_catalog(fil, minsep=10, minmag=None, maxmag=None, outfile=None,
      minstar=-1, Nmax=None, field=None, RAd=None, DECd=None, verbose=False):
   '''Make an intial local-sequence catalog file.

   Args:
      fts (str/obj):  Name of fits file to analyze
      minsep (float): Minimum separation of sources. 
      minmag (float): minimum magnitude (maximum brightness) to include
      maxmag (float): maximum magnitude (minimum brightness) to include
      minstar (float): minimum star classification (CLASS_STAR) to include
      outfile (str): If not none, output the catalog to specified file.
      field (str):  If specified, the field entry for the database. If None,
                    use OBJECT header
      RA/DEC (float): RA and DEC in degrees for the field (SN). If None,
                    try header keys SNRA,SNDEC. If that fails, set to 0,0
      verbose (bool): Output info to terminal?

   Returns:
      catalog (astropy.table.Table):  SNname, object ID, RA, DEC 
      sexcat (astropy.table.Table): source extraxtor table
      PScat (astropy.table.Table):  PS source table
   '''

   fts = fits.open(fil)
   h = fts[0].header

   wcs = WCS(h)
   test_ra,test_dec = wcs.all_pix2world(1,1,1)
   if test_ra == 1.0 and test_dec == 1.0:
      print('Error: file {} has not WCS!'.format(fil))
      return None

   # Detect stars in the image
   if verbose:
      print("Running sextractor...")
   s = sextractor.SexTractor(fil)
   s.run()
   cat = s.parseCatFile()
   cat = cat[cat['FLAGS'] == 0]   # reject bad flags
   if verbose:
      print("Done!  Found {} sources".format(len(cat)))

   # Now retrieve PS stars for this field
   if verbose:
      print("Querying PanSTARRS source catalog...")
   j,i = fts[0].data.shape[0]/2,fts[0].data.shape[1]/2
   RA,DEC = wcs.all_pix2world(i,j,0)
   if 'SCALE' in fts[0].header:
      radius = max(i,j)*sqrt(2)*fts[0].header['SCALE']/3600
   elif 'CDELT1' in fts[0].header:
      radius = max(i,j)*sqrt(2)*fts[0].header['CDELT1']
   else:
      raise ValueError("No scale found, please set it")
   tab = getPS.getStarCat(float(RA), float(DEC), radius)
   if verbose:
      print("Done!  Retrieved {} PS sources".format(len(tab)))

   # find the matching stars.  Sextractor coordinates count from 1
   RA,DEC = wcs.all_pix2world(cat['X_IMAGE'],cat['Y_IMAGE'],1)
   c1 = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree)
   c2 = SkyCoord(ra=tab['RA']*u.degree, dec=tab['DEC']*u.degree)
   idx,d2d,d3d = c1.match_to_catalog_sky(c2)

   # Take a maximum separation of 1 arc-sec
   gids = d2d < 1.0*u.arcsec
   cat = cat[gids]
   idx = idx[gids]
   tab = tab[idx]
   c1 = c1[gids]
   c2 = c2[idx]


   # Now we hunt for multiple hits
   dups = idx[:,newaxis] == idx[newaxis,:]
   nhits = sum(dups, axis=0)
   gids = nhits == 1

   # Get rid of multiple hits, since they should be less than 1 arc-sec
   cat = cat[gids]
   tab = tab[gids]
   if verbose: 
      print('Found {} sources that match PanSTARRS'.format(len(cat)))

   # Remove pairs of objects that are too close
   dists = c1[newaxis,:].separation(c1[:,newaxis])
   bids = greater(sum(dists < minsep*u.arcsec, axis=0), 1)
   if verbose:
      print('Rejecting {} sources that are too close together'.format(
         sum(bids)))

   cat = cat[~bids]
   tab = tab[~bids]

   # lastly, remove filter on magnitude cuts
   col = fts[0].header['filter'] + "mag"
   if col not in tab:
      print("Warning: filter {} has no PS magnitude, using r-mag for cuts")
      col = 'rmag'

   if minmag is not None:
      gids = greater(tab[col], minmag)
      if verbose:
         print("Rejecting {} sources with mag < {}".format(
            sum(~gids), minmag))
      tab = tab[gids]
      cat = cat[gids]

   if maxmag is not None:
      gids = less(tab[col], maxmag)
      if verbose:
         print("Rejecting {} sources with mag > {}".format(
            sum(~gids), maxmag))
      tab = tab[gids]
      cat = cat[gids]

   if minstar > 0:
      gids = greater(cat['CLASS_STAR'], minstar)
      if verbose:
         print("Rejecting {} sources with CLASS_STAR < {}".format(
            sum(~gids), minstar))
      tab = tab[gids]
      cat = cat[gids]

   # Sort by magnitude
   sids = argsort(tab[col])
   tab = tab[sids]
   cat = cat[sids]

   if Nmax is not None:
      tab = tab[:Nmax]
      cat = cat[:Nmax]

   # Make the output table
   if field is None:  field = fts[0].header['OBJECT']
   RA,DEC = wcs.all_pix2world(cat['X_IMAGE'],cat['Y_IMAGE'],1)
   outtab = Table([[field]*len(cat), arange(len(cat))+1, RA, DEC], 
         names=['field','obj','RA','DEC'])

   if RAd is None or DECd is None:
      if 'SNRA' in fts[0].header and 'SNDEC' in fts[0].header:
         RAd = fts[0].header['SNRA']
         DECd = fts[0].header['SNDEC']
      else:
         print("Warning, field coordinates not found. Set to 0,0")
         RAd = DECd = 0
   outtab.insert_row(0, [field, 0, RAd, DECd])
   outtab['RA'].format = "%12.6f"
   outtab['DEC'].format = "%12.6f"

   if outfile is not None:
      if verbose:
         print("Outputting {} objects to {}".format(len(cat), outfile))
      outtab.write(outfile, format='ascii.fixed_width', delimiter=' ',
            overwrite=True)
   return outtab, cat, tab

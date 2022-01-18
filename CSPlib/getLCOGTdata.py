'''Script for downloading LCOGT data from the science archive. Based on script
written by Nestor Espinoza (nespino@astro.puc.cl).'''

import os
import sys
import argparse
import requests
import datetime
import numpy as np
from astropy.io import fits
from astropy.table import Table
from .filesystem import LCOGTname
from .do_astrometry import do_astrometry
from .config import getconfig

cfg = getconfig()

ftemplate = 'https://archive-api.lco.global/frames/?limit={}&RLEVEL=91&'\
            'start={}&PROPID={}'

def extractFITS(ftsfile):
   '''Extract the flat fits files from the LCOGT "cubes" and give them good
   names'''

   base = os.path.dirname(ftsfile)
   fts = fits.open(ftsfile)
   oname = os.path.join(base,LCOGTname(fts))
   img = fits.HDUList()
   img.append(fits.PrimaryHDU(header=fts[1].header, data=fts[1].data))
   img.writeto(oname, overwrite=True)

   cat = Table(fts[2].data)
   cat.write(oname.replace('.fits','.phot.txt'), format='ascii.fixed_width', 
             delimiter=' ', overwrite=True)

   bpm = fits.HDUList()
   bpm.append(fits.PrimaryHDU(header=fts[3].header, data=fts[3].data))
   bpm.writeto(oname.replace('.fits','_bpm.fits'), overwrite=True)

   sig = fits.HDUList()
   sig.append(fits.PrimaryHDU(header=fts[4].header, data=fts[4].data))
   sig.writeto(oname.replace('.fits','_sigma.fits'), overwrite=True)

   os.unlink(ftsfile)
   os.system('touch {}'.format(ftsfile+".stub"))
   return oname

def downloadLatestFrames(prop, headers, basedir, days=None, obj=None):
   """Download the latest Frames from the past [hours] hours. Good to use in
   a cron job that runs on this timescale. For each extended FITS file, 
   extract the images, photometry and solve the WCS

   Args:
      prop (str):  Proposal ID.
      headers (dict):  Authorization token from Authenticate()
      basedir (str):  Path for download. Downloaded files will be saved at
                      basedir/OBJECT
      days (int):  only download frames from this past number of days. If
                    None, then download all files not currently downloaded.
      obj (str):  Only download frames for this object

   Returns:
      list: list of files downloaded

   """
   if days is not None:
      now = datetime.date.today()
      then = now - datetime.timedelta(days=days)
   else:
      then = datetime.date(2000, 1, 1)

   start = then.strftime('%Y-%m-%d')
   url = ftemplate.format(100, start, prop)
   if obj is not None:
      url += '&OBJECT={}'.format(obj)
   response = requests.get(url, headers=headers)
   data = response.json()

   frames = data['results']
   if len(frames) == 0:  return 0
   count = 0
   while True:
      for frame in frames:
         obj = frame['OBJECT'] 
         if obj[0:2] == "SN":
            obj = obj[2:]
         outpath = os.path.join(basedir, obj)
         if not os.path.isdir(outpath):
            os.mkdir(outpath)
         outfile = os.path.join(basedir, obj, frame['filename'])
         if not os.path.isfile(outfile+".stub"):
            with open(outfile, 'wb') as f:
               f.write(requests.get(frame['url']).content)
               count += 1
            ofile = extractFITS(outfile)
            new = do_astrometry([ofile], replace=True, 
                  other=['--overwrite','-p'], dir=cfg.software.astrometry)
             
      if data.get('next'):
         data = requests.get(data['next'], headers=headers).json()
         frames = data['results']
      else:
         break
   return count


def get_headers_from_token(username, password):
    """
      This function gets an authentication token from the LCO archive.

      Args:
          username (string): User name for LCO archive
          password (string): Password for LCO archive
      Returns:
          dict: LCO authentication token
    """
    # Get LCOGT token:
    response = requests.post('https://archive-api.lco.global/api-token-auth/',
                             data={'username': username,
                                   'password': password}
                             ).json()

    token = response.get('token')

    # Store the Authorization header
    headers = {'Authorization': 'Token ' + token}
    return headers


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Download LCOGT frames for a"\
         " given proposal ID and optionally object name")
   parser.add_argument('prop', help='Proposal ID')
   parser.add_argument('user', help='User name')
   parser.add_argument('passwd', help='Password')
   parser.add_argument('-days', help='Number of days in past to search',
         type=int, default=1)
   parser.add_argument('-obj', help='Optional object name', default=None)
   parser.add_argument('-target', help='Target directory for downloads'\
         ' (default: .)', default='.')
   args = parser.parse_args()

   # Get frame names from starting to ending date:

   headers = get_headers_from_token(args.user, args.passwd)

   downloadLatestFrames(args.prop, headers, basedir, days=args.days, 
         obj=args.obj)

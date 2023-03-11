#!/usr/bin/env python


from astropy.io import ascii,fits
from astropy.wcs import WCS
from astropy.table import join,vstack,Table

from CSPlib.phot import ApPhot,compute_zpt
from CSPlib import database
from CSPlib.tel_specs import getTelIns
from CSPlib import config
from CSPlib.config import getconfig

from matplotlib import pyplot as plt
from astropy.visualization import simple_norm

import argparse
import sys,os
import numpy as np

import warnings


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description="Do aperture photometry")
   parser.add_argument("image", help="list of science images", nargs="+")
   parser.add_argument("-cat", help="Catalog file")
   parser.add_argument("-objcol", help="Object column name in catalog file",
                       default="col2")
   parser.add_argument("-RAcol", help="RA column name in catalog file",
                       default="col3")
   parser.add_argument("-DECcol", help="DEC column name in catalog file",
                       default="col4")
   parser.add_argument("-tel", help="Telescope code", default='SWO')
   parser.add_argument("-ins", help="Insrument code", default='NC')
   parser.add_argument("-snap", help="Aperture number for SN", type=int,
         default=-1)
   parser.add_argument("-o", help="Output SN photometryfile", 
         default="SNphot.dat")
   parser.add_argument("-db", help="Database to query LS coordinates (if no cat)", 
                       default='POISE')
   args = parser.parse_args()

   specs = getTelIns(args.tel, args.ins)
   cfg = config.getconfig()
   Naps = len(cfg.photometry.aps)

   SNrows = []

   for imgfile in args.image:

      aphot = ApPhot(imgfile)
      print('Working on {}'.format(imgfile))

      if args.cat is None:
         cat = database.getLSCoords(aphot.object, db=args.db)
         aphot.loadObjCatalog(table=cat, racol='RA', deccol='DEC', objcol='objID')
      else:
         aphot.loadObjCatalog(filename=args.cat, racol=args.RAcol, 
                              deccol=args.DECcol, objcol=args.objcol)


      aphot.makeApertures(appsizes=cfg.photometry.aps, 
               sky_in=cfg.photometry.skyin, sky_out=cfg.photometry.skyout)
       
      #with warnings.catch_warnings():
      #   warnings.simplefilter("ignore")
      #   aphot.plotCutOuts(xcols=6, ycols=6)
      try:
         phot = aphot.doPhotometry()
      except:
         print('Photometry failed for {}, skipping...'.format(imgfile))
         continue
       
      gids = True
      for i in range(0,Naps):
         gids = gids*~np.isnan(phot['ap{}'.format(i)])
         gids = gids*~np.isnan(phot['ap{}er'.format(i)])
      if not np.sometrue(gids):
         print('All the apertures for {} had problems, skipping...'.format(
            imgfile))
         continue
      
      phot = phot[gids]
      phot.rename_column('OBJ','objID')
      phot.remove_column('id')
      phot.sort('objID')
      phot['xcenter'].info.format = "%.2f"
      phot['ycenter'].info.format = "%.2f"
      # Re-order columns
      cols = ['objID','xcenter','ycenter','msky','mskyer']
      for i in range(Naps):
         cols.append('flux{}'.format(i))
         cols.append('eflux{}'.format(i))
         cols.append('ap{}'.format(i))
         cols.append('ap{}er'.format(i))
      cols += ['flags','fits']
      phot = phot[cols]
      phot = phot.filled(fill_value=-1)
      phot.write(imgfile.replace('.fits','.phot'), 
         format='ascii.fixed_width', delimiter=None, overwrite=True)

      # Name of the final aperture (assumed to be the standard)
      apn = "ap{}".format(len(cfg.photometry.aps)-1)
      apner = "ap{}er".format(len(cfg.photometry.aps)-1)

#!/usr/bin/env python
'''Do the optimal extraction.  Uses the opt_extr module to extract the 
photometry  optimally, as per N. Tayler's algorithm.

The basic idea is to fit a PSF to the bright stars in the field as an
elliptical Gaussian or Moffat.  The median FWHMs and angle are then used to model the
PSF of the star we wish to do photometry on.  However, this is not used to
fit a PSF to the star and extract the flux, rather the PSF is used as a 
weighting mask when summing up the counts.  Therefore, counts near the star
center are heavily weighted compared to the ones in the wings, tapering off
to zero semi-continuously.

The sky is fit by making a histogram of the pixel distribution outside 4 FHWM
of the star, then fitting a skewed gaussian to it.  The skew should take care
of any real FAINT objects that are in the sky annulus.'''

from CSPlib import opt_extr
from CSPlib.tel_specs import getTelIns
import sys,os,string, re
import numpy as np
from CSPlib.npextras import between
from CSPlib.phot import compute_zpt
from argparse import ArgumentParser
from astropy.table import Table,join
from astropy.io import ascii
from imagematch.basis import abasis, svdfit

from matplotlib import pyplot as plt

try:
   from tqdm import tqdm
except:
   tqdm = lambda x: x

def solveSNpos(xin, yin, xfit, yfit, nord=3, wt=None, SNid=0):
   '''Given the input pixel coordinates (xin,yin) of the stars in the field
   and the fittend centroids (xfit,yfit), solve for a nord-order transform,
   finally solving for the position of the SN (xSN,ySN).
   nord = 0 is a special case:  shift plus rotation'''
   if wt is None:
      wt = xin*0 + 1
   wt[SNid] = 0    # Don't fit the postion of the SN
   gids = wt > 0

   wt = np.concatenate([wt,wt])
   basis = abasis(nord, xin, yin, rot=[0,1][nord==0])
   try:
      sol = svdfit(basis*wt[:,np.newaxis], np.concatenate([xfit,yfit])*wt)
   except:
      return -1,xin[SNid],yin[SNid],-1,-1
   ixy = np.add.reduce(sol[np.newaxis,:]*basis, 1)
   Nx = len(np.ravel(xin))
   ix,iy = ixy[:Nx], ixy[Nx:]
   SNx = ix[SNid]
   SNy = iy[SNid]
   xrms = np.sqrt(np.mean(np.power(ix-xfit,2)[gids]))
   yrms = np.sqrt(np.mean(np.power(iy-yfit,2)[gids]))
   return 0,SNx,SNy,xrms,yrms

def compute_mag(flux,error,zmag=-1):
   mag = -2.5*np.log10(flux) + zmag
   dmag = 1.0857*error/flux
   return(mag,dmag)

parser = ArgumentParser(description="Produce optimal aperture extracted"\
      " photometry (Naylor, 1998)")
parser.add_argument("fits", help="Fits files to process", nargs="+",
      metavar="fits-files")
parser.add_argument("-tel", help="Telescope code. Default: SWO", 
      default="SWO")
parser.add_argument("-ins", help="Instrument code. Default: NC", 
      default="NC")
parser.add_argument("-catdir", help="Location of catalog files", 
      default=".")
parser.add_argument("-minsep", help="Minimum separation (in arc-sec) from SN"\
      " to avoid for LS stars", default=15, type=float)
parser.add_argument("-optstar", help="Identify of the star to optimize. "\
      "Default: sky-limited", type=int, default=-1)
parser.add_argument("-outsuff", help="Output photometry suffix. Default: .opt",
      default=".opt")
parser.add_argument("-tmin", help="Minimum exposure time to consider as a "\
      "science image", type=float, default=0)
parser.add_argument("-outpat", help="Output file replace pattern. "\
      "Default: .fits", default=".fits")
parser.add_argument("-snout", help="Output file for SN photometry. "\
      "Default: snphot.dat", default='snphot.dat')
parser.add_argument("-logsuff", help="Logfile suffix for copius output",
      default='.optextr.log')
parser.add_argument("-cliprad", help="Clip radius for for profile mask",
      type=float, default=2.0)
#parser.add_argument("-zmag", help="Zero-point, as a number or header key",
#      default='ZP')
parser.add_argument("-dpos", help="Centroiding limit in pixels", type=float,
      default=3.0)
parser.add_argument("-sndpos", help="Centroiding limit in pixels for SN", 
      type=float, default=0.0)
parser.add_argument("-debug", help="Produce debugging info", 
      action="store_true")

args = parser.parse_args()


try:
   specs = getTelIns(args.tel, args.ins)
except:
   print("Error!  Telescope/Instrument code {}/{} not found in database".format(
      args.tel, args.ins))
   print("Set that up before running this program")
   sys.exit(1)

# This will be turned into a table at the end.
sndata = []

for fil in tqdm(args.fits):
   logfile = fil.replace(args.outpat, args.logsuff)
   logf = open(logfile, "w")
   # Instantiate the optimal extraction
   opt = opt_extr.OptExtrPhot(fil, tel=args.tel, ins=args.ins, logf=logf,
         debug=args.debug)
   if opt.exposure < args.tmin:
      continue
   if opt.debug:
      subdatas = []
      weights = []

   # Get the catalog
   LScatfile = os.path.join(args.catdir, opt.obj_name+"_LS.cat")
   if os.path.isfile(os.path.join(args.catdir, opt.obj_name+".nat")):
      allcatfile = os.path.join(args.catdir, opt.obj_name+".nat")
      opt.log("Using natural LS photometry")
   else:
      allcatfile = os.path.join(args.catdir, opt.obj_name+".cat")
      opt.log("Using standard LS photometry")
   if not os.path.isfile(LScatfile):
      if not os.path.isfile(allcatfile):
         opt.log("Error! Catalog not found for {}({})".format(fil,opt.obj_name))
         continue
      LScatfile = allcatfile
   opt.loadObjCatalog(filename=LScatfile, racol='RA',deccol='DEC', 
         objcol='objID')
   allcat = ascii.read(allcatfile, fill_values=[('...',0)])

   # Start with a guess of FWHM = 1 arcsec
   this_fwhm = 1.0/opt.scale

   opt.log("HEADER INFO")
   opt.log("   gain = {}".format(opt.gain))
   opt.log("   fwhm = {}".format( this_fwhm))
#   opt.log("   zmag = {}".format( zmag))
   opt.log("   JD = {}".format( opt.date))
   opt.log("   filter = {}".format(opt.filter))

   # Now, let's model the PSF.  If psfstar is given, use it, otherwise all but
   #  object 0 (the SN)
   opt.log("PSF-CALC:")
   shape_par,ipsf,nfit, rchi = opt.psf_calc(args.dpos, this_fwhm,
         plotfile=fil.replace('.fits','_psf.png'))
   if shape_par is None:
      opt.log("PSF Fit failed, abort.")
      continue

   (fwhm1,fwhm2) = opt.get_fwhm(shape_par)
   opt.log("   FWHM's = {},{}".format(fwhm1,fwhm2))

   clip = args.cliprad*np.sqrt(fwhm1*fwhm2)
   this_fwhm = np.sqrt(fwhm1*fwhm2)
   # Next, we optimize for the given star (if requested), otherwise use sky-
   # limited.
   if args.optstar > 0 and args.optstar in opt.objs:
      idx = opt.objs.index(optstar)
      opt.log("OPT_EXTR:")
      opt.log("   Optimizing extraction for star {} (index {})".format(
         optstar, idx))
      x = opt.xpsf[idx]
      y = opt.ypsf[idx]
      # Call the extraction, using sky-limit to find the peak flux.
      flux, error, xfit, yfit, xerr, yerr, peak, cflag, sky, skynos_r, rchi = \
         opt.extr(x, y, args.dpos, this_fwhm, clip, shape_par, 0.,0.0,0.0, 0.0)
      if cflag != "None":
         # Extraction failed for some reason, let's just extract for sky-limit
         opt.log("   Warning! Extraction of optstar failed. Reverting to "\
               "sky-limit")
         optnrm = 0.0
      else:
         optnrm = peak/skynos_r**2
   else:
      optnrm = 0.0
      opt.log("   Optimizing extraction for sky-limited case")

   # Now for the main event:  call the extr routine on the objects.  First,
   # with input SN location allowing dpos to vary if asked.
   ress = []
   for i in range(len(opt.objs)):
      ress.append(list(opt.extr(opt.xpsf[i], opt.ypsf[i], args.dpos, this_fwhm,
         clip, shape_par, optnrm, 0.0, 0.0, 0.0)))
      if opt.debug:
         subdatas.append(opt._subdata)
         weights.append(opt._weight)

   # make into a table, for ease
   tab = Table(rows=ress, names=['flux','eflux','xfit','yfit','xerr','yerr',\
                                 'peak','cflag','sky','skynos','rchi'])

   zids = np.nonzero(opt.objs==0)[0]
   if len(zids) == 0:
      opt.log("   SN not located in the field... skipping")
      continue

   SNid = np.nonzero(opt.objs==0)[0][0]
   if args.sndpos == 0:
      # Now we fit for a coordinate transform from input pixels to fit pixels
      var = tab['xerr']**2 + tab['yerr']**2
      wt = np.where(var > 0, np.power(var,-1), 0)
      dists = np.sqrt((opt.xpsf-tab['xfit'])**2 + (opt.ypsf-tab['yfit'])**2)
      # Throw out things that are more than 3 pixels discrepant
      wt = np.where(dists < 3, wt, 0)
      stat,xSN,ySN,xrms,yrms = solveSNpos(opt.xpsf, opt.ypsf, tab['xfit'], 
            tab['yfit'], wt=wt, SNid=SNid)
      if stat < 0:
         # The coordinate solution failed
         opt.log('Coord transform failed!  Using simple offset')
         dx = np.sum(wt*(tab['xfit']-opt.xpsf))/np.sum(wt)
         dy = np.sum(wt*(tab['yfit']-opt.ypsf))/np.sum(wt)
         xSN = opt.xpsf[SNid] + dx
         ySN = opt.ypsf[SNid] + dy
         opt.log('(dx,dy) =  ({},{})'.format(dx, dy))
      else:
         opt.log('Coord transform RMS in x = {}, y = {}'.format(xrms, yrms))
         opt.log('SN pixel coordinates:  ({},{})'.format(xSN, ySN))
         opt.log('(dx,dy) = ({},{})'.format(xSN-opt.xpsf[SNid],
                                            ySN-opt.ypsf[SNid]))

   else:
      xSN = opt.xpsf[SNid]
      ySN = opt.ypsf[SNid]

   res = list(opt.extr(xSN, ySN, args.sndpos, this_fwhm, clip, shape_par, 
      optnrm, 0.0, 0.0, 0.0))
   if opt.debug:
      subdatas.append(opt._subdata)
      weights.append(opt._weight)
   opt.log("   Supernova extracted with:")
   opt.log("      flux = {} +/- {}".format(res[0],res[1]))
   opt.log("      xfit = {} +/- {}".format(res[2],res[4]))
   opt.log("      yfit = {} +/- {}".format(res[3],res[5]))
   opt.log("      rchi2 = {}".format(res[-1]))
   if res[0] <= 0:  res[7] = 'Z'    # flux < 0

   idx = np.nonzero(opt.objs==0)[0][0]
   # Update with SN values
   tab.remove_row(idx)
   tab.insert_row(0, res)

   tab['objID'] = opt.objs

   mag,emag = compute_mag(tab['flux'],tab['eflux'], zmag=30)
   mag = np.where(tab['flux'] > 0, mag, 99)
   emag = np.where(tab['flux'] > 0, emag, -1)
   tab['magins'] = mag
   tab['emagins'] = emag
   tab['filt'] = opt.filter
   tab['JD'] = opt.date

   # Join with the standards catalog and solve for a zp
   if opt.filter+'mag' in allcat.colnames:
      mkey = opt.filter + 'mag'
      ekey = opt.filter + 'err'
   else:
      mkey = opt.filter
      ekey = 'e'+opt.filter
     
   if mkey not in allcat.colnames or ekey not in allcat.colnames:
      tab['mag'] = -1
      tab['emag'] = -1
      tab['mstd'] = 0
      tab['emstd'] = 0

   else:
      tab = join(tab, allcat['objID',mkey,ekey], keys='objID')
      tab.rename_column(mkey,'mstd')
      tab.rename_column(ekey,'emstd')
      if hasattr(tab['mstd'], 'mask'):
         tab = tab[~tab['mstd'].mask]

      zp1,ezp1,flags1,mesg1 = compute_zpt(tab, 'mstd','emstd','magins','emagins',
            zpins=0)
      zp2,ezp2,flags2,mesg2 = compute_zpt(tab, 'mstd','emstd','magins','emagins',
            zpins=0, plot=fil.replace(args.outpat, "_zp.png"), use_pymc=True)

      if zp1 is None:
         print("Failed to compute a zero-point for {}".format(fil))
         print("Message is {}".format(mesg))
         tab['mag1'] = -1
         tab['emag1'] = -1
         tab['mag2'] = -1
         tab['emag2'] = -1
      else:
         tab['mag1'] = tab['magins'] + zp1
         tab['emag1'] = np.sqrt(tab['emagins']**2 + ezp1**2)
         tab['mag2'] = tab['magins'] + zp2
         tab['emag2'] = np.sqrt(tab['emagins']**2 + ezp2**2)

      for col in ['flux','eflux','xfit','yfit','xerr','yerr','peak','skynos',
            'rchi','magins','emagins','mstd','emstd','mag1','emag1','mag2',
            'emag2']:
         tab[col].info.format = "%.4f"
      tab = tab['objID','filt','JD','xfit','yfit','xerr','yerr','flux',
            'eflux','peak', 'cflag','sky','skynos','rchi','magins','emagins',
            'mstd','emstd', 'mag1','emag1','mag2','emag2']

      # We've got good magnitudes to get the SN data
      idx = np.nonzero(tab['objID'] == 0)[0][0]
      if tab['mag1'][idx] > 0:
         sndata.append([opt.obj_name, opt.filter, opt.date, 
            tab['mag1'][idx], tab['emag1'][idx],
            tab['mag2'][idx], tab['emag2'][idx],fil])

         ## Now, let's try to get a reliable measure of the disersions
         ## First get rid of the outliers
         #gids = np.greater(tab['objID'], 0)*np.less(tab['magins'], 90)
         #for i in range(3):
         #   med = np.median(tab[gids]['mstd']-tab[gids]['mag'])
         #   mad = 1.49*np.median(np.absolute(tab[gids]['mstd']-\
         #         tab[gids]['mag']-med))
         #   gids = gids*np.less(np.absolute(tab['mstd']-tab['mag']), 5*mad)
         ## next consider the 20 objects with mag closest to the SN
         #ids = np.argsort(tab[gids]['mag']-tab[idx]['mag'])[:20]
         #shift = np.mean(tab[gids][ids]['mstd']-tab[gids][ids]['mag'])
         #stddev = np.std(tab[gids][ids]['mstd']-tab[gids][ids]['mag']-shift)
         #sndata[-1] = sndata[-1] + [mad,shift,stddev]

   tab.write(fil.replace(args.outpat, args.outsuff), format='ascii.fixed_width',
         delimiter=' ', overwrite=True)
   if opt.debug:
      nfigs = len(subdatas)//49+1
      for j in range(nfigs):
         isize = weights[0].shape[0]
         jsize = weights[0].shape[1]
         print(isize,jsize)
         imdata = np.zeros((7*isize, 7*jsize))
         wdata = np.zeros((7*isize, 7*jsize))
         fig,ax = plt.subplots(figsize=(6,6))
         for k in range(7):
            for l in range(7):
               if j*49+k*7+l > len(subdatas)-1: break
               imdata[k*isize:(k+1)*isize,l*jsize:(l+1)*jsize] = \
                     subdatas[j*49+k*7+l]/subdatas[j*49+k*7+l].max()
               wdata[k*isize:(k+1)*isize,l*jsize:(l+1)*jsize] = \
                     weights[j*49+k*7+l]
         ax.imshow(imdata)
         ax.contour(wdata, levels=[0.25,0.75], colors='red', linewidth=1)
         fig.savefig(fil.replace('.fits','.psf{}.png'.format(j)))



sntab = Table(rows=sndata, 
      names=['SN','filter','JD','mag1','emag1','mag2','emag2','fits'])
sntab['JD'].info.format = "%.5f"
sntab['mag1'].info.format = "%.4f"
sntab['emag1'].info.format = "%.4f"
sntab['mag2'].info.format = "%.4f"
sntab['emag2'].info.format = "%.4f"
sntab.write(args.snout, format='ascii.fixed_width', delimiter=' ', 
      overwrite=True)

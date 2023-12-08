'''This module contains a pipeline class that does all the organizational
work of classifying images types, do the calibrations, and watching for
new files.'''

import matplotlib
matplotlib.use('Agg')
from astropy.io import fits,ascii
from astropy.coordinates import SkyCoord
from astropy.time import Time
import datetime
from astropy import units as u
from astropy.wcs import WCS
from astropy import table
from astropy.stats import sigma_clipped_stats
from .npextras import between
import numpy as np
from .phot import ApPhot, compute_zpt, PSFPhot
from .calibration import getOptNaturalMag
from . import ccdred
from . import headers
from . import do_astrometry
from . import opt_extr
from .fitsutils import qdump
from .filesystem import CSPname
from imagematch import ImageMatching_scalerot as ImageMatch
from .objmatch import WCStoImage
import os, subprocess
from os.path import join,basename,isfile,dirname,isdir
from glob import glob
import time
import re
import signal
from . import database
from .config import getconfig

from matplotlib import pyplot as plt

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

cfg = getconfig()

def night2JD(night):
   '''Given a night code (e.g., ut210427_28), give a JD'''
   res = re.search('ut([0-9][0-9])([0-9][0-9])([0-9][0-9])_([0-9][0-9])',night)
   if res is not None:
      yy,mm,dd1,dd2 = res.groups()
      dt = datetime.datetime(2000+int(yy), int(mm), int(dd1), 12,0,0)
      return Time(dt).jd
   else:
      return None

def closestNight(jd, nights):
   '''Given a list of nights, find the night closest in time.
   nights are in the form utyymmdd_dd'''
   jds = np.array([night2JD(n) for n in nights])
   return nights[np.argmin(np.absolute(jds-jd))]

filtlist = cfg.data.filtlist
sex_dir = join(dirname(__file__), 'data', 'sex')

stopped = False

class Pipeline:

   def __init__(self, datadir, workdir=None, prefix='ccd', suffix='.fits',
         calibrations=cfg.data.calibrations, templates=cfg.data.templates,
         catalogs=cfg.data.templates, fsize=9512640, tmin=0, update_db=True,
         gsub=None, reduced=None, SNphot=cfg.photometry.SNphot):
      '''
      Initialize the pipeline object.

      Args:
         datadir (str):  location where the data resides
         workdir (str):  location where the pipeline will do its work. If
                         None, same as datadir
         prefix/suffix (str):  Prefix  and suffix for raw data. 
                         glob(prefix+'*'+suffix) should return all files.
         fsize (int):  Expected size of the CCD file. If it's not this size
                       (plus n*2880), we skip (it could still be reading out)
         calibrations (str): location where older calibration files are located
         templates (str): location where templates are stored
         catalogs (str): location where catalog files are stored
         tmin (float):  Only FITS files with exposure times > tmin are 
                        reduced (to avoid calibration images).
         update_db (bool):  If True, update the CSP database with the 
                            SN photometry.
         gsub (str): location where galaxy subtraction images should be
                     stored. If None, the work folder.
         reduced (str): location where reduced (bias-subtraced, flat-fielded,
                        and WCS computed files are stored. If None, they are
                        only left in the working folder.
         SNphot (str): File where supernova photometry will be saved to file.
      Returns:
         Pipeline object
      '''

      if not isdir(datadir):
         raise FileNotFoundError("Error, datadir {} not found. Abort!".format(
            datadir))
      self.datadir = datadir
      self.prefix = prefix
      self.suffix = suffix
      self.fsize = [fsize+i*2880 for i in range(3)]  # that should be enough!
      self.tmin = tmin
      self.update_db = update_db

      # A list of all files we've dealt with so far
      self.rawfiles = []
      # A list of bad raw ccd files we want to ingore
      self.badfiles = []
      # A list of files that have been bias-corrected
      self.bfiles = []
      # A list of files that have been flat-fielded
      self.ffiles = []
      # The ZTF designation for each identified object, indexed by ccd frame
      self.ZIDs = {}
      # The standards
      self.stdIDs = {}
      # These are files that are flagged as not needing template subtractions
      self.skipTemplate = []
      # These are files that are not identified or failed in some other way
      self.ignore = []
      # These are files that have short exposures and are likely "test"
      self.short = []
      # These objects have no galaxy template (u-band usually)
      self.no_temp = []
      # These are files that have WCS solved
      self.wcsSolved = []
      # These are files with initial photometry
      self.initialPhot = []
      # Files that have been template-subtracted and had SN photometry done
      self.subtracted = []
      # Files that have final Photometry
      self.finalPhot = []
      # SN photometry saved here
      self.SNphot = SNphot

      # Cache information so we don't have to open/close FITS files too often
      self.headerData = {}

      if workdir is None:
         self.workdir = self.datadir
      else:
         if not isdir(workdir):
            try:
               os.makedirs(workdir)
            except:
               raise OSError(
               "Cannot create workdir {}. Permission problem? Aborting".format(
                  workdir))
         self.workdir = workdir

      # Where calibrations are saved
      if calibrations is None:
         self.calibrations = self.workdir
      else:
         if not isdir(calibrations):
            raise OSError("No such calibration folder: {}".format(calibrations))
         self.calibrations = calibrations
      
      if templates is None:
         self.templates = self.workdir
      else:
         if not isdir(templates):
            raise OSError("No such templates folder: {}".format(templates))
         self.templates = templates

      if gsub is not None:
         if not isdir(gsub):
            raise OSError("No such gsub folder: {}".format(gsub))
         self.gsub = gsub
      else:
         self.gsub = None

      if reduced is not None:
         if not isdir(reduced):
            try:
               os.mkdir(reduced)
            except:
               raise OSError("Cannot create reduced folder: {}".format(reduced))
         self.reduced = reduced
      else:
         self.reduced = None

      try:
         self.logfile = open(join(workdir, "pipeline.log"), 'a')
      except:
         raise OSError("Can't write to workdir {}. Check permissions!".format(
            workdir))

      # Lists of types of files
      self.files = {
            'dflat':{},  # dome flats
            'sflat':{},  # Sky flats
            'astro':{},  # astronomical objects of interest
            'zero':[],   # bias frames
            'none':[],   # ignored
      }
      for filt in filtlist: 
         self.files['dflat'][filt] = []
         self.files['sflat'][filt] = []
         self.files['astro'][filt] = []

      self.biasFrame = None
      self.shutterFrames = {}
      self.flatFrame = {}
      for filt in filtlist:
         self.flatFrame[filt] = None


   def log(self, message):
      '''log the message to the log file and print it to the screen.'''
      print(message)
      self.logfile.write(message+"\n")
      self.logfile.flush()

   def Rclone(self, location, target='.'):
      '''Use rclone to get a file specified by "location" and put it
      in "target".'''
      cmd = "{} copy {} {}".format(cfg.software.rclone, location, target)
      res = os.system(cmd)
      return res

   def addFile(self, filename):
      '''Add a new file to the pipeline. We need to do some initial fixing
      of header info, then figure out what kind of file it is, then add
      it to the queue.'''
      if not isfile(filename):
         self.log("File {} not found. Did it disappear?".format(filename))
         self.badfiles.append(filename)
         return

      # Update header
      fout = join(self.workdir, basename(filename))
      if isfile(fout):
         fts = fits.open(fout, memmap=False)
      else:
         try:
            fts = headers.update_header(filename, fout)
         except:
            self.log('Warning: had a problem with the headers for {}, '\
                     'skipping...'.format(filename))
            self.badfiles.append(filename)
            return

      fil = basename(filename)
      self.headerData[fil] = {}
      for h in ['OBJECT','OBSTYPE','FILTER','EXPTIME','OPAMP','RA','DEC']:
         self.headerData[fil][h] = fts[0].header[h]

      # Figure out what kind of file we're dealing with
      obstype = fts[0].header['OBSTYPE']
      obj = fts[0].header['OBJECT']
      filt = fts[0].header['FILTER']
      if obstype not in headers.obstypes:
         self.log("Warning!  File {} has unrecognized OBSTYPE {}".format(
            filename,obstype))
         self.rawfiles.append(filename)
         self.files['none'].append(fout)
         return
      obtype = headers.obstypes[obstype]
      # Ignore:  will still be calibrated to 'fcd', but no final photometry
      if obtype == 'astro' and fts[0].header['EXPTIME'] < self.tmin:
         self.log("File {} is considered 'short', no SN photometry will be "\
                  "done".format(fout))
         # Flag short exposures
         self.short.append(fout)

      self.rawfiles.append(filename)
      if obtype in ['zero','none']:
         self.files[obtype].append(fout)
      else:
         if filt in self.files[obtype]:
            self.files[obtype][filt].append(fout)
         else:
            self.files['none'].append(fout)

      self.log("New file {} added to queue, is of type {}".format(fout,obtype))

   def getNewFiles(self):
      '''Get a list of files we've not dealt with yet.'''
      flist = glob(join(self.datadir, "{}*{}".format(
         self.prefix, self.suffix)))
      flist.sort()

      new = [f for f in flist if not os.path.islink(f)]
      new = [f for f in new if os.path.getsize(f) in self.fsize]
      new = [f for f in new if f not in self.rawfiles+self.badfiles]

      return new

   def makeFileName(self, fil, suffix=".fits"):
      '''Given the file fil, determine a filename with prefix
      and suffix using info in the header. It will have the 
      format like  SN2010aaa_B01_SWO_NC_2021_02_02.fits'''

      #template = "{obj}_{filt}{idx:02d}_{tel}_{ins}_{YY}_{MM}_{DD}{suf}"
      #fts = fits.open(fil)
      #dateobs = fts[0].header['DATE-OBS']
      #YY,MM,DD = dateobs.split('-')
      #args = dict(YY=YY, MM=MM, DD=DD,
      #            obj=fts[0].header['OBJECT'],
      #            filt=fts[0].header['FILTER'],
      #            tel=fts[0].header['TELESCOP'],
      #            ins=fts[0].header['INSTRUM'],suf=suffix,idx=1)
      #fts.close()
      idx = 1
      while(isfile(CSPname(fil, idx, suffix))): idx += 1
      
      return CSPname(fil, idx, suffix)

   def getHeaderData(self, fil, key):
      f = basename(fil)
      f = 'c'+f[1:]
      return self.headerData[f][key]

   def getWorkName(self, fil, prefix):
      '''Add a prefix to the filename in the work folder.'''
      fil = basename(fil)
      return join(self.workdir, prefix+fil[1:])

   def getDataName(self, fil):
      '''Given a working file name, get the raw name it came from'''
      fil = basename(fil)
      return join(self.datadir, 'c'+fil[1:])

   def makeBias(self):
      '''Make BIAS frame from the data, or retrieve from other sources.'''
      # Can we make a bias frame?
      bfile = join(self.workdir, 'Zero{}'.format(self.suffix))
      if isfile(bfile):
         self.biasFrame = fits.open(bfile, memmap=False)
         self.log("Found existing BIAS: {}, using that".format(bfile))
         return
      if len(self.files['zero']) :
         self.log("Found {} bias frames, building an average...".format(
            len(self.files['zero'])))
         self.biasFrame = ccdred.makeBiasFrame(self.files['zero'], 
               outfile=bfile)
         self.log("BIAS frame saved to {}".format(bfile))
      else:
         # We need a backup BIAS frame
         res = self.Rclone(
               "CSP:Swope/Calibrations/latest/Zero{}".format(self.suffix),
               self.workdir)
         if res == 0:
            self.log("Retrieved BIAS frame from latest reductions")
            self.biasFrame = fits.open(join(self.workdir, 
               'Zero{}'.format(self.suffix)), memmap=False)
         else:
            cfile = join(self.calibrations, "CAL", "Zero{}".format(
               self.suffix))
            self.biasFrame = fits.open(cfile)
            self.biasFrame.writeto(bfile, output_verify='ignore')
            self.log("Retrieved backup BIAS frame from {}".format(cfile))

   def makeFlats(self):
      '''Make flat Frames from the data or retrieve from backup sources.'''

      for filt in filtlist:
         fname = join(self.workdir, "SFlat{}{}".format(filt,
               self.suffix))
         if isfile(fname):
             self.flatFrame[filt] = fits.open(fname, memmap=False)
             self.log("Found existing flat {}. Using that.".format(fname))
             continue
         if len(self.files['sflat'][filt]) > 3:
            self.log("Found {} {}-band sky flats, bias and flux correcting..."\
                  .format(len(self.files['sflat'][filt]), filt))
            files = [self.getWorkName(f,'b') for f in self.files['sflat'][filt]]
            self.flatFrame[filt] = ccdred.makeFlatFrame(files, outfile=fname)
            self.log("Flat field saved to {}".format(fname))
         else:
            # Find the best flat based on date
            # First, we need the current date JD
            fts = fits.open(self.rawfiles[0])
            # We don't have a JD in the header yet, so use EPOCH
            jd = Time(fts[0].header['EPOCH'], format='decimalyear').jd

            # Now get list of Flats we have in the database
            cmd = [cfg.software.rclone, 'ls','--include',
                   'ut*/SFlat{}{}'.format(filt,self.suffix),
                   'CSP:Swope/Calibrations']

            res = subprocess.run(cmd, stdout=subprocess.PIPE)
            lines = res.stdout.decode('utf-8').split('\n')
            lines = [line for line in lines if len(line) > 0]
            flats = [line.split()[1] for line in lines]
            if len(flats) > 0:
               # Pick the closest
               flat = closestNight(jd, flats)
               res = self.Rclone("CSP:Swope/Calibrations/{}".format(flat),
                        self.workdir)
               if res == 0:
                  self.log("Retrieved Flat SFlat{}{} from latest reductions".\
                        format(filt,self.suffix))
                  self.flatFrame[filt] = fits.open(fname, memmap=False)
            else:
               cfile = join(self.calibrations, "CAL", 
                     "SFlat{}{}".format(filt, self.suffix))
               self.flatFrame[filt] = fits.open(cfile, memmap=False)
               self.flatFrame[filt].writeto(fname)
               self.log("Retrieved backup FLAT frame from {}".format(cfile))

   def BiasLinShutCorr(self):
      '''Do bias, linearity, and shutter corrections to all files except bias 
      frames.'''
      if self.biasFrame is None:
         self.log('Abort due to lack of bias frame')
         raise RuntimeError("Error:  can't proceed without a bias frame!")
      todo = []
      for f in self.rawfiles:
         base = basename(f)
         wfile = self.getWorkName(f, 'c')
         bfile = self.getWorkName(f, 'b')
         if wfile not in self.files['zero'] and bfile not in self.bfiles:
            if isfile(bfile):
               self.bfiles.append(bfile)
               continue
            todo.append(wfile)

      for f in todo:
         self.log('Bias correcting CCD frames...')
         fts = ccdred.biasCorrect(f, overscan=True, frame=self.biasFrame)
         err = ccdred.makeSigmaMap(fts)
         # Get the correct shutter file
         opamp = self.getHeaderData(f,'OPAMP')
         if opamp not in self.shutterFrames:
            shfile = join(self.calibrations, 'CAL', "SH{}.fits".format(opamp))
            self.shutterFrames[opamp] = fits.open(shfile, memmap=False)
         fts = ccdred.LinearityCorrect(fts)
         err = ccdred.LinearityCorrect(fts, sigma=err)
         fts = ccdred.ShutterCorrect(fts, frame=self.shutterFrames[opamp])
         err = ccdred.ShutterCorrect(err, frame=self.shutterFrames[opamp])
         bfile = self.getWorkName(f, 'b')
         fts.writeto(bfile, overwrite=True)
         err.writeto(bfile.replace('.fits','_sigma.fits'), overwrite=True)
         self.bfiles.append(bfile)
         self.log('   Corrected file saved to {}'.format(bfile))

   def FlatCorr(self):
      '''Do flat field correction to all files that need it:  astro basically
      '''
      # Process files in 'astro' type that we haven't done yet and that have
      # been bias-corrected
      todo = []
      for filt in self.files['astro']:
         for f in self.files['astro'][filt]:
            bfile = self.getWorkName(f, 'b')
            ffile = self.getWorkName(f, 'f')
            if bfile in self.bfiles and ffile not in self.ffiles:  
               if isfile(ffile):
                  self.ffiles.append(ffile)
                  continue
               todo.append(bfile)

      for f in todo:
         filt = self.getHeaderData(f,'FILTER')
         bfile = self.getWorkName(f, 'b')
         ffile = self.getWorkName(f, 'f')
         sfile1 = bfile.replace('.fits','_sigma.fits')
         sfile2 = ffile.replace('.fits','_sigma.fits')
         if filt not in self.flatFrame:
            raise RuntimeError("No flat for filter {}. Abort!".format(filt))
         self.log("Flat field correcting {} --> {}".format(bfile,ffile))
         fts = ccdred.flatCorrect(bfile, self.flatFrame[filt],
               outfile=ffile)
         # Copying should be fine, since flat is scaled to have mode = 1.0
         os.system('cp {} {}'.format(sfile1,sfile2))
         self.ffiles.append(ffile)

   def identify(self):
      '''Figure out the identities of the objects and get their data if
      we can.'''
      todo = [f for f in self.ffiles if f not in self.ignore]
      for f in todo:
         if f in self.ZIDs or f in self.stdIDs:  continue   # done it already
         filt = self.getHeaderData(f, 'FILTER')
         obj = self.getHeaderData(f,'OBJECT')
         self.log("OBJECT is {}, FILTER = {}".format(obj,filt))

         # First, if this is a standard, and keep in different list
         if obj.find('CSF') == 0 or obj.find('PS') == 0:
            self.log("This is a standard star field")
            ref = join(self.templates, "{}_r.fits".format(obj))
            if not isfile(ref):
               ret = self.Rclone('CSP:Swope/templates/{}_r.fits'.format(obj),
                     self.templates)
               if ret != 0:
                  self.log("Can't get reference image from gdrive. skipping")
                  self.ignore.append(f)
                  continue
            # All's good, we'll consider it
            self.stdIDs[f] = obj

         else:
            # First, check to see if the catalog exists locally
            catfile = join(self.templates, obj+'.nat')
            if isfile(catfile):
               self.ZIDs[f] = obj
            else:
               # Next, try to lookup csp2 database
               res = database.getNameCoords(obj)
               if res == -2:
                  self.log('Could not contact csp2 database, trying gdrive...')
                  ret = self.Rclone('CSP:Swope/templates/{}.nat'.format(obj),
                        self.templates)
                  if ret != 0:
                     self.log("Can't contact csp2 or gdrive, giving up!")
                     self.ignore.append(f)
                     continue
                  else:
                     self.ZIDs[f] = obj
               elif res == -1:
                  self.log('Object {} not found in database, trying coords'.\
                        format(obj))
                  ra = self.getHeaderData(f,'RA')
                  dec = self.getHeaderData(f,'DEC')
                  c = SkyCoord(ra, dec, unit=(u.hourangle, u.degree))
                  res = database.getCoordsName(c.ra.value, c.dec.value)
                  if res == -1 or res == -2:
                     self.log('Coordinate lookup failed, assuming other...')
                     self.ignore.append(f)
                     continue
         
                  self.log('Found {} {} degrees from frame center'.format(
                     res[0], res[3]))
                  ra,dec = res[1],res[2]
                  self.ZIDs[f] = res[0]
               else:
                  self.ZIDs[f] = obj

            # At this point, self.ZIDS[f] is the ZTF ID
            if filt in ['B','V']:
               tmpname = "{}_g.fits".format(self.ZIDs[f])
            else:
               tmpname = "{}_{}.fits".format(self.ZIDs[f], filt)

            skipf = "{}_tskip".format(self.ZIDs[f])
            # check if template exists, or if we have a skip directive
            if isfile(join(self.templates, skipf)):
               self.skipTemplate.append(f)
            elif not isfile(join(self.templates, tmpname)) and filt =='u':
               self.skipTemplate.append(f)
            elif not isfile(join(self.templates, tmpname)):
               res = self.Rclone('CSP:Swope/templates/{}'.format(tmpname),
                     self.templates)
               if res == 0:
                   self.log('Retrieved template file {}'.format(tmpname))
               else:
                   self.log('Failed to get template from gdrive: {}'.format(
                       tmpname))
                   self.no_temp.append(f)
                   continue

            # Get the catalog file
            catfile = "{}.nat".format(self.ZIDs[f])
            if not isfile(join(self.templates, catfile)):
               res = self.Rclone('CSP:Swope/templates/{}'.format(catfile),
                     self.templates)
               if res == 0:
                   self.log('Retrieved catalog file {}'.format(catfile))
               else:
                   self.log('Failed to get catalog file from gdrive: {}'.format(
                       catfile))
                   self.ignore.append(f)
                   continue
            tab = ascii.read(join(self.templates,"{}.nat".format(self.ZIDs[f])))
            if 0 not in tab['objID']:
               self.log('No SN object in catalog file, skipping...')
               self.ignore.append(f)
               continue
            else:
               # Update FITS header with SN position
               idx = list(tab['objID']).index(0)
               ra = tab[idx]['RA']
               dec = tab[idx]['DEC']
               fts = fits.open(f,memmap=False)
               fts[0].header['SNRA'] = "{:.6f}d".format(ra)
               fts[0].header['SNDEC'] = "{:.6f}d".format(dec)
               fts.writeto(f, overwrite=True)

   def solve_wcs(self):
      '''Go through the astro files and solve for the WCS. This can go
      one of two ways:  either we get a quick solution from catalog
      matching, or if that fails, use astrometry.net (slower).'''
      todo = [fil for fil in list(self.ZIDs.keys())+list(self.stdIDs.keys()) \
            if fil not in self.wcsSolved and fil not in self.ignore]

      for fil in todo:
         self.log("Solving WCS for {}".format(fil))
         if fil in self.ZIDs:
            ZID = self.ZIDs[fil]
            standard = False
         else:
            ZID = self.stdIDs[fil]
            standard = True
         filt = self.getHeaderData(fil, 'FILTER')
      
         # check to see if we have a wcs already
         fts = fits.open(fil, memmap=False)
         wcs = WCS(fts[0])
         if wcs.has_celestial:
            self.wcsSolved.append(fil)
            fts.close()
            continue

         # Now, we need to rotate 90 degrees to match up with the sky
         if 'ROTANG' not in fts[0].header:
            fts[0].data = fts[0].data.T
            fts[0].data = fts[0].data[:,::-1]
            fts[0].header['ROTANG'] = 90
            fts.writeto(fil, overwrite=True)
            if isfile(fil.replace('.fits','_sigma.fits')):
               # Do the same transformation to the noise map
               fts = fits.open(fil.replace('.fits','_sigma.fits'))
               fts[0].data = fts[0].data.T
               fts[0].data = fts[0].data[:,::-1]
               fts[0].header['ROTANG'] = 90
               fts.writeto(fil.replace('.fits','_sigma.fits'), overwrite=True)

         fts.close()

         if standard:
            wcsimage = join(self.templates, "{}_r.fits".format(
               ZID,filt))
         else:
            if filt in ['u','B','V']:
               wcsimage = join(self.templates, "{}_{}.fits".format(
                  ZID,'g'))
            else:
               wcsimage = join(self.templates, "{}_{}.fits".format(
                  ZID,filt))
         if os.path.isfile(wcsimage):
            h = fits.getheader(wcsimage)
            if 'TELESCOP' not in h or h['TELESCOP'] != 'SkyMapper':
               #try:
               new = WCStoImage(wcsimage, fil, angles=[0],
                        plotfile=fil.replace('.fits','_wcs.png'))
               #except:
               #   # Something went wrong
               #   new = None
            else:
               new = None
         else:
               new = None
         if new is None:
            self.log("Fast WCS failed... resorting to astrometry.net")
            new = do_astrometry.do_astrometry([fil], replace=True,
                  verbose=True, other=['--overwrite','-p'], 
                  dir=cfg.software.astrometry)
            if new is None:
               self.log("astrometry.net failed for {}. No WCS coputed, "
                        "skipping...".format(fil))
               self.ignore.append(fil)
               continue
            else:
               self.wcsSolved.append(fil)
         else:
            new.writeto(fil, overwrite=True)
            self.wcsSolved.append(fil)

         if self.reduced is not None:
            # Make a copy in the reduced area
            os.system('cp {} {}'.format(fil, self.reduced))

      return

   def photometry(self, bgsubtract=False, crfix=False, computeFWHM=True):
      '''Using the PanSTARRS catalog, we do initial photometry on the field
      and determine a zero-point. Or, if we have stanard fields, we do
      the aperture photometry on them and determine a zero-point.
      
      Args:
          bgsubtract (bool):  If True, do a 2D background estimate.
          crfix (bool):       If True, fix cosmic rays using LAcosmic
          computeFWHM (bool): If True, compute FWHM and update header'''

      todo = [fil for fil in self.wcsSolved if fil not in self.initialPhot \
            and fil not in self.ignore]

      # If we have standards, keep a record for zero-pointing later
      if len(list(self.stdIDs.keys())) > 0:
         if not (isfile('standards.phot')):
            stdf = open('standards.phot','w')
            stdf.write('{:12s} {:2s} {:7s} {:6s} {:7s} {:6s} '\
                 '{:5s} {:6s} {:s}\n'.format('Field','filt','mins','emins',
                    'mag','emag','airm','expt','fits'))
         else:
            stdf = open('standards.phot','a')
      else:
         stdf = None

      for fil in todo:
         standard = fil in self.stdIDs
         self.log('Working on photometry for {}'.format(fil))
         # Check to see if we've done the photometry already
         if isfile(fil.replace('.fits','.phot0')):
            self.initialPhot.append(fil)
            continue
         filt = self.getHeaderData(fil, 'FILTER')
         if not standard:
            obj = self.ZIDs[fil]
            catfile = join(self.templates, '{}_LS.cat'.format(obj))
            allcat = ascii.read(join(self.templates, '{}.nat'.format(obj)),
                  fill_values=[('...',0)])
         else:
            obj = self.stdIDs[fil]
            catfile = join(self.templates, '{}_LS.cat'.format(obj))
            allcat = getOptNaturalMag(filt)
            allcat.rename_column('OBJ','objID')

         if not isfile(catfile):
            if standard:
               # It should be there. Maybe standard name misspelled?
               self.log("Unrecognized standard name {}, maybe "\
                     "misspelled? Skipping...".format(obj))
               self.ignore.append(fil)
               continue
            # First, see if we can retrieve it:
            res = self.Rclone('CSP:Swope/templates/{}_LS.cat'.format(obj),
               self.templates)
            if res > 0:
               # Now remove stars below/above thresholds
               gids = np.ones(len(allcat), dtype=bool)
               for filt in ['u','g','r','i','B','V']:
                  m = getattr(allcat[filt], 'mask', np.zeros(len(allcat), dtype=bool))
                  gids = gids*(~m)
               allcat = allcat[gids]
               gids = np.array(allcat['r'] < 20)
               #gids = gids*(allcat['r'] > 12)
               gids = gids*np.array(np.greater(allcat['er'], 0))
               # make sure well-separated
               ra = np.array(allcat['RA']);  dec = np.array(allcat['DEC'])
               maxdist = 11.
               dists = np.sqrt(np.power(dec[np.newaxis,:]-dec[:,np.newaxis],2)\
                     + np.power((ra[np.newaxis,:]-ra[:,np.newaxis])*\
                     np.cos(dec*np.pi/180), 2))
               Nnn = np.sum(np.less(dists, 11.0/3600), axis=0)
               gids = gids*np.equal(Nnn,1)
               # Check for no more than 400 objects
               while sum(gids) > 200:
                  maxdist += 1
                  Nnn = np.sum(np.less(dists, maxdist/3600), axis=0)
                  gids = gids*np.equal(Nnn,1)
               if 0 in allcat['objID']:
                  # make sure SN is kept!
                  idx = list(allcat['objID']).index(0)
                  gids[idx] = True
               cat = allcat[gids]
               cat = cat['objID','RA','DEC']
               cat['id'] = np.arange(len(cat))
               cat['RA'].info.format="%10.6f"
               cat['DEC'].info.format="%10.6f"
               self.log('Creating LS catalog with {} objets'.format(len(cat)))
               cat.write(catfile, format='ascii.fixed_width', delimiter=' ')
            else:
               cat = ascii.read(catfile)
         else:
            cat = ascii.read(catfile)

         ap = ApPhot(fil, sigma=fil.replace('.fits','_sigma.fits'))
         ap.loadObjCatalog(table=cat, racol='RA', deccol='DEC', 
               objcol='objID')

         update = False     # Do we need to update the image file?
         if not standard:
            self.log("Doing Cosmic ray fix")
            try:
               ap.CRReject(fix=crfix, sigclip=5)
               if crfix: 
                  update = True
               qdump(fil.replace('.fits','_bpm.fits'), ap.mask.astype(np.int8), fil)
            except:
               pass
         
         if not standard:
            self.log("Modeling 2D background")
            try:
               ap.model2DBackground(boxsize=100)
               qdump(fil.replace('.fits','_bg.fits'),
                     ap.background.background.astype(ap.data.dtype), fil)
               if bgsubtract: 
                  update = True
                  ap.head['MEANSKY'] = np.round(ap.background.background_median, 3)
                  ap.data = ap.data - ap.background.background
            except:
               pass

         if computeFWHM and not standard:
            fwhm,tab = ap.fitFWHM(plotfile=fil.replace('.fits','_fwhm.pdf'), 
                                  profile='Gauss')
            if fwhm > 0:
               ap.head['FWHM'] = np.round(fwhm,3)
               update = True
         
         if update:
            qdump(fil, ap.data, ap.head)

         self.log('Doing aperture photometry...')
         try:
            phot = ap.doPhotometry()
         except:
            self.log("Doing aperture photometry failed for {}, "\
                  "skipping...".format(fil))
            self.ignore.append(fil)
            continue
         phot.rename_column('OBJ','objID')
         if not standard:
            phot = table.join(phot, allcat['objID',filt,'e'+filt], keys='objID')
            phot.rename_column(filt,filt+'mag')
            phot.rename_column('e'+filt, filt+'err')

            #phot = table.join(phot, allcat['objID',filt+'mag',filt+'err'],
            #      keys='objID')
            phot[filt+'mag'].info.format='%.4f'
            phot[filt+'err'].info.format='%.4f'
         else:
            phot = table.join(phot, allcat['objID','mag','emag'], keys='objID')
            phot['mag'].info.format='%.4f'
            phot['emag'].info.format='%.4f'

         phot.remove_column('id')

         # Just the good stuff
         gids = (~np.isnan(phot['ap2er']))*(~np.isnan(phot['ap2']))
         if not np.any(gids):
            self.log("Initial photomery failed for {}, skipping...".format(
               fil))
            self.ignore.append(fil)
            continue
         phot = phot[gids]
         phot.sort('objID')
         phot['exptime'] = self.getHeaderData(fil, 'EXPTIME')
         
         phot.write(fil.replace('.fits','.phot0'), format='ascii.fixed_width',
               delimiter=' ', fill_values=[(ascii.masked, '...')])
         if standard:
            # We're done for now.
            self.initialPhot.append(fil)

            # Update standard photomery
            for i in range(len(phot)):
               if phot['mag'].mask[i]: continue
               if phot['flags'][i] > 0: continue
               stdf.write('{:12s} {:.2s} {:7.4f} {:6.4f} {:7.4f} {:6.4f} '\
                 '{:5.3f} {:6.1f} {:s}\n'.format(obj,filt, 
                 *phot[i]['ap2','ap2er','mag','emag','airmass','exptime',
                          'fits']))

            continue
         gids = np.greater(phot['objID'], 0)
         if hasattr(phot[filt+'mag'], 'mask'): 
            gids = gids*(~phot[filt+'mag'].mask)
         gids = gids*between(phot[filt+'mag'], 15, 20)
         if not np.any(gids):
            self.log("Determining zero-point for frame {} failed, "\
                  "skipping...".format(fil))
            self.ignore.append(fil)
            continue
         phot = phot[gids]
         diffs = phot[filt+'mag']- phot['ap2']
         mn,md,st = sigma_clipped_stats(diffs, sigma=3)

         # throw out 5-sigma outliers with respect to MAD
         mad = 1.5*np.median(np.absolute(diffs - md))
         gids = np.less(np.absolute(diffs - md), 5*mad)
         if not np.any(gids):
            self.log("Determining zero-point for frame {} failed, "\
                  "skipping...".format(fil))
            self.ignore.append(fil)
            continue

         # Weight by inverse variance
         wts = np.power(phot['ap2er']**2 + phot[filt+'err']**2,-1)*gids

         # 30 is used internall in photometry code as arbitrary zero-point
         zp = np.sum(diffs*wts)/np.sum(wts) + 30
         ezp = np.sqrt(1.0/np.sum(wts))
         #zp = md + 30
         #ezp = st/np.sqrt(sum(gids))
         if np.isnan(zp) or np.isnan(ezp):
            self.log("Determining zero-point for frame {} failed (NAN), "\
                  "skipping...".format(fil))
            self.ignore.append(fil)
            continue

         self.log('Determined zero-point to be {} +/- {}'.format(zp,ezp))
         fts = fits.open(fil, memmap=False)
         fts[0].header['ZP'] = zp
         fts[0].header['EZP'] = ezp

         # make some diagnostic plots of aperture correction and zp determ.
         fig,axes = plt.subplots(2,1, figsize=(6,6))
         diffs = phot[filt+'mag']- phot['ap2']
         x = phot[filt+'mag']
         y = diffs + 30
         axes[0].errorbar(x, y, fmt='o', xerr=phot[filt+'err'], 
               yerr=np.sqrt(phot[filt+'err']**2 + phot['ap2er']**2))
         axes[0].plot(x[~gids],y[~gids], 'o', mfc='red', label='rejected',
               zorder=10)
         axes[0].axhline(zp, color='k')
         axes[0].set_xlim(12,20)
         axes[0].set_ylim(zp-1,zp+1)
         axes[0].legend()
         axes[0].set_ylabel('m(std) - m(ins)')
         axes[0].set_xlabel('m(std)')
         self.initialPhot.append(fil)

         # Now aperture corrections
         for i,r in [('0',3.0),('1',5.0)]:
            ap = 'ap'+i
            aper = 'ap'+i+'er'
            gids = (~np.isnan(phot[ap]))*(~np.isnan(phot[aper]))*\
                  (np.greater(phot['objID'], 0))
            diffs = np.where(gids, phot['ap2']-phot[ap], 0)
            wts = np.where(gids,np.power(phot['ap2er']**2+phot[aper]**2,-1), 0)
            apcor = np.sum(wts*diffs)/np.sum(wts)
            eapcor = np.sqrt(1.0/np.sum(wts))
            self.log('   Aperture correction 2 -> {} is {:.3f}'.format(
               i,apcor))
            fts[0].header['APCOR2'+i] = apcor
            fts[0].header['EAPCOR2'+i] = eapcor
            xs = np.linspace(r-0.25,r+0.25, sum(gids))
            axes[1].errorbar(xs, diffs[gids], yerr=np.power(wts[gids],-0.5),
                  fmt='o')
            axes[1].errorbar([r], [apcor], fmt='o', yerr=[eapcor],
                  color='red')

         axes[1].axhline(0, color='k', zorder=100)
         axes[1].set_xlabel('apsize (arc-sec) + random')
         axes[1].set_ylabel('mag(7") - mag(ap)')
         axes[1].set_ylim(-1,1)
         fts[0].writeto(fil, overwrite=True)
         fig.savefig(fil.replace('.fits','_zp.jpg'))
      if stdf is not None: stdf.close()


      return

   def subPSFPhotometry(self):
      '''Using the PanSTARRS or SkyMapper catalogs, perform PSF photometry
      on the difference images.'''

      todo = [fil for fil in self.subtracted if \
            fil not in self.finalPhot and fil not in self.ignore]

      for fil in todo:
         self.log('Working on final PSF photometry for {}'.format(fil))
         if isfile(fil.replace('.fits','.psf')):
            try:
               test = ascii.read(fil.replace('.fits','.psf'))
               self.finalPhot.append(fil)
               continue
            except:
               pass
         obj = self.ZIDs[fil]
         filt = self.getHeaderData(fil, 'FILTER')
         catfile = join(self.templates, '{}_LS.cat'.format(obj))
         cat = ascii.read(catfile)
         if 'id' not in cat.colnames:
            cat['id'] = np.arange(len(cat))
         allcat = ascii.read(join(self.templates, '{}.nat'.format(obj)),
                  fill_values=[('...',0)])

         psf = PSFPhot(fil.replace('.fits','diff.fits'), tel='SWO', ins='NC')
         # Use 'id' instead of 'objID' as MAGINS can't handle the large ints
         psf.loadObjCatalog(table=cat, racol='RA', deccol='DEC',
               objcol='id')
         try:
            tab = psf.doPhotometry(magins=cfg.software.magins, 
                  stdcat=cfg.data.stdcat)
         except:
            self.log("PSF Photometry failed... skipping")
            self.ignore.append(fil)
            continue
         # Bring objID back in to the table
         tab.rename_column('OBJ','id')
         tab = table.join(tab, cat['id','objID'], keys='id')
         
         # SN index
         zids = np.nonzero(psf.objs==0)[0]
         if len(zids) == 0:
            self.log("PSF:   SN not located in field... skipping")
            self.ignore.append(fil)
            continue

         #tab.rename_column('OBJ','objID')
         tab.rename_column('mag1','magins')
         tab.rename_column('merr1','emagins')
         tab.rename_column('filter','filt')
         tab.rename_column('date','JD')

         # Join with the standards catalog and solve for a zp
         if psf.filter+'mag' in allcat.colnames:
            mkey = psf.filter + 'mag'
            ekey = psf.filter + 'err'
         else:
            mkey = psf.filter
            ekey = 'e'+psf.filter
           
         if mkey not in allcat.colnames or ekey not in allcat.colnames:
            self.log("Standard magnitude {} not found in catalog. Skipping...".\
                  format(mkey))
            self.ignore.append(fil)
            continue

         tab = table.join(tab, allcat['objID',mkey,ekey], keys='objID')
         tab.rename_column(mkey,'mstd')
         tab.rename_column(ekey,'emstd')
         if hasattr(tab['mstd'], 'mask'):
            tab = tab[~tab['mstd'].mask]
  
         zp,ezp,flags,mesg = compute_zpt(tab,'mstd','emstd','magins',
               'emagins', zpins=0)
  
         if zp is None:
            self.log("Failed to compute a zero-point for {}".format(fil))
            self.log("Message is {}".format(mesg))
            self.ignore.append(fil)
            continue
         tab['mag'] = tab['magins'] + zp
         tab['emag'] = np.sqrt(tab['emagins']**2 + ezp**2)
         tab['eflux'] = tab['emagins']*tab['flux']/1.087
         tab['eflux'].info.format = "%.4f"
  
         tab = tab['objID','filt','JD','xc','yc','flux',
               'eflux','msky','mskyer','chi','magins','emagins',
               'mstd','emstd', 'mag','emag','flags','perr','g1','g2']
  
         # We've got good magnitudes to get the SN data
         if 0 not in tab['objID']:
            self.log("No PSF photometry for the SN, skipping...")
            self.ignore.append(fil)
            continue
         idx = list(tab['objID']).index(0)
         if tab['mag'][idx] <= 0:
            self.log("SN data for {} invalid, skipping...".format(fil))
            self.ignore.append(fil)
            continue
         mag = tab['mag'][idx]
         emag = tab['emag'][idx]
         with open(self.SNphot, 'a') as fout:
            fout.write("{:20s} {:2s} {:.3f} {:.3f} {:.3f} {}\n".format(
                       obj, filt, psf.date, mag, emag, 
                       basename(fil.replace('.fits','diff.fits'))))
  
         tab.write(fil.replace('.fits','.psf'), format='ascii.fixed_width',
               delimiter=' ', overwrite=True)

         if self.update_db:
            self.log("Updating CSP database with photometry for {},{}".format(
               obj,filt))
            res = database.updateSNPhot(obj, jd, filt, basename(fil), mag, emag)
            if res == -2:
               self.log('Failed to udpate csp2 database')
         self.finalPhot.append(fil)
      return


   def subOptPhotometry(self):
      '''Using the PanSTARRS or SkyMapper catalogs, permfor optimized photometry
      on the difference images.'''

      todo = [fil for fil in self.subtracted if \
            fil not in self.finalPhot and fil not in self.ignore]

      for fil in todo:
         self.log('Working on final optimized photometry for {}'.format(fil))
         if isfile(fil.replace('.fits','.opt')):
            try:
               test = ascii.read(fil.replace('.fits','.opt'))
               self.finalPhot.append(fil)
               continue
            except:
               pass
         obj = self.ZIDs[fil]
         filt = self.getHeaderData(fil, 'FILTER')
         catfile = join(self.templates, '{}_LS.cat'.format(obj))
         cat = ascii.read(catfile)
         allcat = ascii.read(join(self.templates, '{}.nat'.format(obj)),
                  fill_values=[('...',0)])

         logf = open(fil.replace('.fits','opt.log'), 'w')
         opt = opt_extr.OptExtrPhot(fil, tel='SWO', ins='NC',
               logf=logf)

         opt.loadObjCatalog(filename=catfile, racol='RA', deccol='DEC',
               objcol='objID')
         this_fwhm = 1.0/opt.scale

         opt.log("HEADER INFO")
         opt.log("   gain = {}".format(opt.gain))
         opt.log("   fwhm = {}".format( this_fwhm))
         opt.log("   JD = {}".format( opt.date))
         opt.log("   filter = {}".format(opt.filter))
  
         # Now, let's model the PSF.  If psfstar is given, use it, otherwise 
         # all but object 0 (the SN)
         opt.log("PSF-CALC:")
         shape_par,ipsf,nfit, rchi = opt.psf_calc(3.0, this_fwhm,
               plotfile=fil.replace('.fits','_psf.png'))
         if shape_par is None:
            self.log("OPT:   PSF Fit failed, abort.")
            self.ignore.append(fil)
            continue

         (fwhm1,fwhm2) = opt.get_fwhm(shape_par)
         self.log("OPT:    FWHM's = {},{}".format(fwhm1,fwhm2))
  
         clip = 2.0*np.sqrt(fwhm1*fwhm2)
         this_fwhm = np.sqrt(fwhm1*fwhm2)

         # results
         ress = []
         for i in range(len(opt.objs)):
            ress.append(list(opt.extr(opt.xpsf[i], opt.ypsf[i], 3.0, this_fwhm,
               clip, shape_par, 0.0, 0.0, 0.0, 0.0)))
            if ress[-1][0] <= 0:  ress[-1][7] = 'Z'     # flux <= 0
         # make into a table, for ease
         tab = table.Table(rows=ress,names=['flux','eflux','xfit','yfit','xerr',
                                       'yerr','peak','cflag','sky','skynos',
                                       'rchi'])

         # SN index
         zids = np.nonzero(opt.objs==0)[0]
         if len(zids) == 0:
            self.log("OPT:   SN not located in field... skipping")
            self.ignore.append(fil)
            continue

         SNid = np.nonzero(opt.objs==0)[0][0]
         xSN = opt.xpsf[SNid]
         ySN = opt.ypsf[SNid]

         res = list(opt.extr(xSN, ySN, 3.0, this_fwhm, clip, shape_par, 
            0.0, 0.0, 0.0, 0.0))
         self.log("OPT:    Supernova extracted with:")
         self.log("OPT:       flux = {} +/- {}".format(res[0],res[1]))
         self.log("OPT:       xfit = {} +/- {}".format(res[2],res[4]))
         self.log("OPT:       yfit = {} +/- {}".format(res[3],res[5]))
         self.log("OPT:       rchi2 = {}".format(res[-1]))
         if res[0] <= 0:  res[7] = 'Z'    # flux < 0
  
         idx = np.nonzero(opt.objs==0)[0][0]
         # Update with SN values
         tab.remove_row(idx)
         tab.insert_row(0, res)
  
         tab['objID'] = opt.objs

         # Catch the zero/negative fluxes
         mag = np.where(tab['flux'] > 0, -2.5*np.log10(tab['flux']) + 30, -1)
         emag = np.where(tab['flux'] > 0, 1.0857*tab['eflux']/tab['flux'], -1)

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
            self.log("Standard magnitude {} not found in catalog. Skipping...".\
                  format(mkey))
            self.ignore.append(fil)
            continue

         tab = table.join(tab, allcat['objID',mkey,ekey], keys='objID')
         tab.rename_column(mkey,'mstd')
         tab.rename_column(ekey,'emstd')
         if hasattr(tab['mstd'], 'mask'):
            tab = tab[~tab['mstd'].mask]
         tab['flags'] = np.where(tab['cflag'] == 'O', 0, 1)
  
         zp,ezp,flags,mesg = compute_zpt(tab,'mstd','emstd','magins',
               'emagins', zpins=0)
  
         if zp is None:
            self.log("Failed to compute a zero-point for {}".format(fil))
            self.log("Message is {}".format(mesg))
            self.ignore.append(fil)
            continue
         tab['mag'] = tab['magins'] + zp
         tab['emag'] = np.sqrt(tab['emagins']**2 + ezp**2)
  
         # Format the table
         for col in ['flux','eflux','xfit','yfit','xerr','yerr','peak','skynos',
               'rchi','magins','emagins','mstd','emstd','mag','emag']:
            tab[col].info.format = "%.4f"
         tab = tab['objID','filt','JD','xfit','yfit','xerr','yerr','flux',
               'eflux','peak', 'cflag','sky','skynos','rchi','magins','emagins',
               'mstd','emstd', 'mag','emag','flags']
  
         # We've got good magnitudes to get the SN data
         idx = np.nonzero(tab['objID'] == 0)[0][0]
         if tab['mag'][idx] <= 0:
            self.log("SN data for {} invalid, skipping...".format(fil))
            self.ignore.append(fil)
            continue
         mag = tab['mag'][idx]
         emag = tab['emag'][idx]
         with open(self.SNphot, 'a') as fout:
            fout.write("{:20s} {:2s} {:.3f} {:.3f} {:.3f} {}\n".format(
                       obj, filt, opt.date, mag, emag, basename(fil)))
  
         tab.write(fil.replace('.fits','.opt'), format='ascii.fixed_width',
               delimiter=' ', overwrite=True)

         if self.update_db:
            self.log("Updating CSP database with photometry for {},{}".format(
               obj,filt))
            res = database.updateSNPhot(obj, opt.date, filt, basename(fil), 
                  mag, emag)
            if res == -2:
               self.log('Failed to udpate csp2 database')
         self.finalPhot.append(fil)
      return


   def subphotometry(self):
      '''Using the PanSTARRS catalog, we do subtracted photometry on the field
      and update the database.'''
 
      todo = [fil for fil in self.subtracted if \
            fil not in self.finalPhot and fil not in self.ignore]


      for fil in todo:
         self.log('Working on final photometry for {}'.format(fil))
         if isfile(fil.replace('.fits','.phot')):
            try:
               test = ascii.read(fil.replace('.fits','.phot'))
               self.finalPhot.append(fil)
               continue
            except:
               pass
         obj = self.ZIDs[fil]
         filt = self.getHeaderData(fil, 'FILTER')
         catfile = join(self.templates, '{}_LS.cat'.format(obj))
         cat = ascii.read(catfile)
         allcat = ascii.read(join(self.templates, '{}.nat'.format(obj)),
                  fill_values=[('...',0)])
         fts = fits.open(fil, memmap=False)
         if 'ZP' not in fts[0].header:
            self.log('No zero-point. Skipping...')
            self.ignore.append(fil)
            continue
         zpt = fts[0].header['ZP']
         ezpt = fts[0].header['EZP']
         apcor = fts[0].header['APCOR20']
         jd = fts[0].header['JD']
         fts.close()

         ap = ApPhot(fil.replace('.fits','diff.fits'))
         ap.loadObjCatalog(table=cat, racol='RA', deccol='DEC', 
               objcol='objID')
         self.log('Doing aperture photometry...')
         phot = ap.doPhotometry()
         phot.rename_column('OBJ','objID')
         phot = table.join(phot, allcat['objID',filt,'e'+filt],
               keys='objID')
         phot.remove_column('id')
         phot.rename_column(filt,filt+'mag')
         phot.rename_column('e'+filt,filt+'err')

         # Just the good stuff
         gids = (~np.isnan(phot['ap2er']))*(~np.isnan(phot['ap2']))
         phot = phot[gids]
         phot.sort('objID')
         phot.write(fil.replace('.fits','.phot'), format='ascii.fixed_width',
               delimiter=' ', fill_values=[(ascii.masked, '...')])
         if 0 not in phot['objID']:
            self.log("object photometry failed for {}, skipping...".format(
               fil))
            self.ignore.append(fil)
            continue
         idx = list(phot['objID']).index(0)
         if phot[idx]['flux0'] < 0:
            mag = 25
            emag = 1.0
         else:
            mag = phot[idx]['ap0'] - 30 + zpt + apcor
            emag = np.sqrt(phot[idx]['ap0er']**2 + ezpt**2)
         with open(self.SNphot, 'a') as fout:
            fout.write("{:20s} {:2s} {:.3f} {:.3f} {:.3f} {}\n".format(
               obj, filt, jd, mag, emag, basename(fil)))
         if self.update_db:
            self.log("Updating CSP database with photometry for {},{}".format(
               obj,filt))
            res = database.updateSNPhot(obj, jd, filt, basename(fil), mag, emag)
            if res == -2:
               self.log('Failed to udpate csp2 database')
         self.finalPhot.append(fil)
      return

   def template_subtract(self):
      '''For objects with initial photometry, do template-subtraction
      and then redo the photometry for the SN object'''

      todo = [fil for fil in self.initialPhot if fil not in self.subtracted \
            and fil not in self.ignore and fil not in self.stdIDs and \
            fil not in self.no_temp and fil not in self.short]
      for fil in todo:
         obj = self.ZIDs[fil]
         diff = fil.replace('.fits','diff.fits')

         # Check to see if we skip template subtractions for this file
         if fil in self.skipTemplate:
            # Just copy the file
            self.log("No template needed, skipping subtraction")
            os.system("cp {} {}".format(fil,diff))
            self.subtracted.append(fil)

         magcat = join(self.templates, "{}.nat".format(obj))
         # Check to see if we've done it already
         if isfile(diff): 
            self.subtracted.append(fil)
            continue

         filt = self.getHeaderData(fil, 'FILTER')
         if filt in ['B','V']:
            stemplate = join(self.templates, '{}_g.fits'.format(obj,filt))
         else:
            stemplate = join(self.templates, '{}_{}.fits'.format(obj,filt))

         # If the template is missing, assume we don't do it
         if cfg.data.copyTemplates:
            template = join(self.workdir, os.path.basename(stemplate))
            if not os.path.isfile(template):
               os.system("cp {} {}".format(stemplate, template))
         else:
            template = stemplate

         obs = ImageMatch.Observation(fil, scale=0.435, saturate=4e4, 
               reject=True, snx='SNRA', sny='SNDEC', magmax=22,
               magmin=11)
         ref = ImageMatch.Observation(template, scale=0.25, saturate=6e4,
               reject=True, magmax=22, magmin=11)
         try:
            res = obs.GoCatGo(ref, skyoff=True, pwid=11, perr=3.0, nmax=100, 
                  nord=3, match=True, subt=True, quick_convolve=True, 
                  do_sex=True, thresh=3., sexdir=sex_dir, diff_size=35,bs=False,
                  usewcs=True, xwin=[200,1848], ywin=[200,1848], vcut=1e8,
                  magcat=magcat, magcol='r')
            if res != 0:
               self.log('Template subtraction failed for {}, skipping'.format(
                     fil))
               self.ignore.append(fil)
               continue
            self.subtracted.append(fil)

            # If requested, save the template subtraction image
            if self.gsub is not None:
               subimg = fil.replace('.fits','SN_diff.jpg')
               if os.path.isfile(subimg):
                  newf = self.makeFileName(fil, 'SN_diff.jpg')
                  os.system('cp {} {}'.format(subimg, newf))
                  # where to put it
                  sdir = os.path.join(self.gsub, obj)
                  if not os.path.isdir(sdir):
                     os.mkdir(sdir)
                  os.system('cp {} {}'.format(newf, sdir))
         except:
            self.log('Template subtraction failed for {}, skipping'.format(
                fil))
            self.ignore.append(fil)

   def initialize(self):
      '''Make a first run through the data and see if we have what we need
      to get going. We can always fall back on generic calibrations if
      needed.'''

      self.log("Start pipeline at {}".format(
          time.strftime('%Y/%m/%d %H:%M:%S')))
      files = self.getNewFiles()
      for fil in files:
         self.addFile(fil)

      self.makeBias()
      self.BiasLinShutCorr()
      self.makeFlats()
      self.FlatCorr()

   def sighandler(self, sig, frame):
      if sig == signal.SIGHUP:
         self.stopped = True


   def run(self, poll_interval=10, wait_for_write=2):
      '''Check for new files and process them as they come in. Only check
      when idle for poll_interval seconds. Also, we wait wait_for_write
      number of seconds before reading each file to avoid race conditions.'''
      self.stopped = False
      signal.signal(signal.SIGHUP, self.sighandler)
      done = False
      while not self.stopped:
         if poll_interval > 0: 
            time.sleep(poll_interval)
         else:
            if done: break
         #print("Checking for new files")
         files = self.getNewFiles()
         for fil in files:
            time.sleep(wait_for_write)
            self.addFile(fil)

         self.BiasLinShutCorr()
         self.FlatCorr()

         self.identify()
         if not cfg.tasks.WCS:
            done = True
            continue
         self.solve_wcs()

         if not cfg.tasks.InitPhot:
            done = True
            continue
         self.photometry(bgsubtract=False, crfix=True, computeFWHM=True)

         if not cfg.tasks.TempSubt:
            done = True
            continue
         self.template_subtract()

         if not cfg.tasks.SubPhot:
            done = True
            continue
         if cfg.photometry.instype == 'optimal':
            self.subOptPhotometry()
         elif cfg.photometry.instype == 'psf':
            self.subPSFPhotometry()
         else:
            self.subphotometry()

      self.log("Pipeline stopped normally at {}".format(
         time.strftime('%Y/%m/%d %H:%M:%S')))
      return



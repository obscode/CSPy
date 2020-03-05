'''This module contains a pipeline class that does all the organizational
work of classifying images types, do the calibrations, and watching for
new files.'''

from astropy.io import fits,ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import table
import numpy as np
from .phot import ApPhot
from . import ccdred
from . import headers
from . import do_astrometry
from .objmatch import WCStoImage
import os
from os.path import join,basename,isfile,dirname,isdir
from glob import glob
import time
import signal
from . import database

filtlist = ['u','g','r','i','B','V']
calibrations_folder = '/csp21/csp2/software/SWONC'

stopped = False

class Pipeline:

   def __init__(self, datadir, workdir=None, prefix='ccd', suffix='.fits'):
      '''
      Initialize the pipeline object.

      Args:
         datadir (str):  location where the data resides
         workdir (str):  location where the pipeline will do its work. If
                         None, same as datadir
         prefix/suffix (str):  Prefix  and suffix for raw data. 
                         glob(prefix+'*'+suffix) should return all files.
         poll_interval(int): Sleep for this many seconds between checks for
                        new files
      Returns:
         Pipeline object
      '''

      if not isdir(datadir):
         raise FileNotFoundError("Error, datadir {} not found. Abort!".format(
            datadir))
      self.datadir = datadir
      self.prefix = prefix
      self.suffix = suffix

      # A list of all files we've dealt with so far
      self.rawfiles = []
      # A list of files that have been bias-corrected
      self.bfiles = []
      # A list of files that have been flat-fielded
      self.ffiles = []
      # The ZTF designation for each identified object, indexed by ccd frame
      self.ZIDs = {}
      # These are files that are not identified or failed in some other way
      self.ignore = []
      # These are files that have WCS solved
      self.wcsSolved = []
      # These are files with initial photometry
      self.initialPhot = []
      # Files that have been template-subtracted and had SN photometry done
      self.subracted = []

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


      try:
         self.logfile = open(join(workdir, "pipeline.log"), 'w')
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

   def addFile(self, filename):
      '''Add a new file to the pipeline. We need to do some initial fixing
      of header info, then figure out what kind of file it is, then add
      it to the queue.'''
      if not isfile(filename):
         self.log("File {} not found. Did it disappear?".format(filename))
         return

      # Update header
      fout = join(self.workdir, basename(filename))
      if isfile(fout):
         fts = fits.open(fout)
      else:
         fts = headers.update_header(filename, fout)

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

      new = [f for f in flist if f not in self.rawfiles]

      return new

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
         self.biasFrame = fits.open(bfile)
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
         cmd = 'rclone copy CSP:Swope/Calibrations/latest/Zero{} {}'.format(
               self.suffix, self.workdir)
         res = os.system(cmd)
         if res == 0:
            self.log("Retrieved BIAS frame from latest reductions")
            self.biasFrame = fits.open(join(self.workdir, 
               'Zero{}'.format(self.suffix)))
         else:
            cfile = join(calibrations_folder, "CAL", "Zero{}".format(
               self.suffix))
            self.biasFrame = fts.open(cfile)
            self.biasFrame.writeto(bfile)
            self.log("Retrieved backup BIAS frame from {}".format(cfile))

   def makeFlats(self):
      '''Make flat Frames from the data or retrieve from backup sources.'''

      for filt in filtlist:
         fname = join(self.workdir, "SFlat{}{}".format(filt,
               self.suffix))
         if isfile(fname):
             self.flatFrame[filt] = fits.open(fname)
             self.log("Found existing flat {}. Using that.".format(fname))
             continue
         if len(self.files['sflat'][filt]) > 3:
            self.log("Found {} {}-band sky flats, bias and flux correcting..."\
                  .format(len(self.files['sflat'][filt]), filt))
            files = [self.getWorkName(f,'b') for f in self.files['sflat'][filt]]
            self.flatFrame[filt] = ccdred.makeFlatFrame(files, outfile=fname)
            self.log("Flat field saved to {}".format(fname))
         else:
            cmd = "rclone copy CSP:Swope/Calibrations/latest/SFlat{}{} {}".\
                  format(filt,self.suffix,self.workdir)
            ret = os.system(cmd)
            if ret == 0:
               self.log("Retrieved Flat SFlat{}{} from latest reductions".\
                     format(filt,self.suffix))
               self.flatFrame[filt] = fits.open(fname)
            else:
               cfile = join(calibrations_folder, "CAL", 
                     "SFlat{}{}".format(filt, self.suffix))
               self.flatFrame[filt] = fits.open(cfile)
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
         fts = ccdred.biasCorrect(f, overscan=True, frame=self.biasFrame)
         # Get the correct shutter file
         opamp = self.getHeaderData(f,'OPAMP')
         if opamp not in self.shutterFrames:
            shfile = join(calibrations_folder, 'CAL',
                  "SH{}.fits".format(opamp))
            self.shutterFrames[opamp] = fits.open(shfile)
         fts = ccdred.LinearityCorrect(fts)
         fts = ccdred.ShutterCorrect(fts, frame=self.shutterFrames[opamp])
         bfile = self.getWorkName(f, 'b')
         fts.writeto(bfile, overwrite=True)
         self.bfiles.append(bfile)

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
         if filt not in self.flatFrame:
            raise RuntimeError("No flat for filter {}. Abort!".format(filt))
         self.log("Flat field correcting {} --> {}".format(bfile,ffile))
         fts = ccdred.flatCorrect(bfile, self.flatFrame[filt],
               outfile=ffile)
         self.ffiles.append(ffile)

   def identify(self):
      '''Figure out the identities of the objects and get their data if
      we can.'''
      for f in self.ffiles:
         if f in self.ZIDs:  continue   # done it already
         dname = self.getDataName(f)
         filt = self.getHeaderData(f, 'FILTER')
         if filt not in ['g','r','i']:
            self.log('Skipping {} with filter {}'.format(f,filt))
            self.ignore.append(f)
         if f not in self.ZIDs and f not in self.ignore:
            obj = self.getHeaderData(f,'OBJECT')

            # First, check to see if the catalog exists locally
            catfile = join(self.workdir, obj+'.cat')
            if isfile(catfile):
               self.ZIDs[f] = obj
            else:
               # Next, try to lookup csp2 database
               res = database.getNameCoords(obj, db='LCO')
               if res == -2:
                  self.log('Could not contact csp2 database, trying gdrive...')
                  cmd = 'rclone ls CSP:Swope/templates/{}.cat'.format(obj)
                  ret = os.system(cmd)
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
                  res = database.getCoordsName(c.ra.value, d.dec.value, 
                        db='LCO')
                  if res == -1 or res == -2:
                     self.log('Coordinate lookup failed, assuming standard...')
                     self.ignore.append(f)
                     continue
 
                  self.log('Found {} {} degrees from frame center'.format(
                     res[0], res[3]))
                  ra,dec = res[1],res[2]
                  self.ZIDs[f] = res[0]
               else:
                  self.ZIDs[f] = res[0]
         # At this point, self.ZIDS[f] is the ZTF ID
         tmpname = "{}_{}.fits".format(self.ZIDs[f], filt)
         if not isfile(join(self.workdir, tmpname)):
            cmd = 'rclone copy CSP:Swope/templates/{}_{}.fits {}'.format(
                    self.ZIDs[f], filt, self.workdir)
            res = os.system(cmd)
            if res == 0:
                self.log('Retrieved template file {}'.format(tmpname))
            else:
                self.log('Failed to get template from gdrive: {}'.format(
                    tmpname))
                self.ignore.append(f)
         # Get the catalog file
         catfile = "{}.cat".format(self.ZIDs[f])
         if not isfile(join(self.workdir, catfile)):
            cmd = 'rclone copy CSP:Swope/templates/{} {}'.format(
                    catfile, self.workdir)
            res = os.system(cmd)
            if res == 0:
                self.log('Retrieved catalog file {}'.format(catfile))
            else:
                self.log('Failed to get catalog file from gdrive: {}.'.format(
                    catfile))

   def solve_wcs(self):
      '''Go through the astro files and solve for the WCS. This can go
      one of two ways:  either we get a quick solution from catalog
      matching, or if that fails, use astrometry.net (slower).'''
      todo = [fil for fil in self.ZIDs if fil not in self.wcsSolved]

      for fil in todo:
         ZID = self.ZIDs[fil]
         filt = self.getHeaderData(fil, 'FILTER')
         wcsimage = join(self.workdir, "{}_{}.fits".format(
            ZID,filt))
         new = WCStoImage(wcsimage, fil, angles=np.arange(-2,2.5,0.5))
         if new is None:
            self.log("Fast WCS failed... resorting to astrometry.net")
            new = do_astrometry.do_astrometry([fil], replace=True,
                  dir='/usr/local/', verbose=True, other=['--overwrite'])
            if new is None:
               self.log("astrometry.net failed for {}. No WCS coputed, "
                        "skipping...".format(fil))
               self.ignore.append(fil)
            else:
               self.wcsSolved.append(fil)
         else:
            self.wcsSolved.append(fil)
      return

   def photometry(self):
      '''Using the PanSTARRS catalog, we do initial photometry on the field
      and determine a zero-point.'''

      todo = [fil for fil in self.wcsSolved if fil not in self.initialPhot]

      for fil in todo:
         self.log('Working on photometry for {}'.format(fil))
         obj = self.getHeaderData(fil, 'OBJECT')
         filt = self.getHeaderData(fil, 'FILTER')
         catfile = join(self.workdir, '{}_LS.cat'.format(obj))
         allcat = ascii.read(join(self.workdir, '{}.cat'.format(obj)))
         if not isfile(catfile):
            # Now remove stars below/above thresholds
            gids = allcat['rmag'] < 20
            gids = gids*(allcat['rmag'] > 12)
            gids = gids*np.greater(allcat['rerr'], 0)
            # make sure well-separated
            ra = allcat['RA'];  dec = allcat['DEC']
            dists = np.sqrt(np.power(dec[np.newaxis,:]-dec[:,np.newaxis],2) +\
               np.power((ra[np.newaxis,:]-ra[:,np.newaxis])*\
               np.cos(dec*np.pi/180), 2))
            Nnn = np.sum(np.less(dists, 11.0/3600), axis=0)
            gids = gids*np.equal(Nnn,1)
            cat = allcat[gids]
            cat = cat['objID','RA','DEC']
            cat['RA'].info.format="%10.6f"
            cat['DEC'].info.format="%10.6f"
            self.log('Creating LS catalog with {} objets'.format(len(cat)))
            cat.write(catfile, format='ascii.fixed_width', delimiter=' ')
         else:
            cat = ascii.read(catfile)

         ap = ApPhot(fil)
         ap.loadObjCatalog(table=cat, racol='RA', deccol='DEC', 
               objcol='objID')
         self.log('Doing aperture photometry...')
         phot = ap.doPhotometry()
         phot.rename_column('OBJ','objID')
         phot = table.join(phot, allcat['objID',filt+'mag',filt+'err'],
               keys='objID')

         phot.write(fil.replace('.fits','.phot'), format='ascii.fixed_width',
               delimiter=' ')
         gids = (~np.isnan(phot['ap2er']))*(~np.isnan(phot['ap2']))
         diffs = np.where(gids, phot[filt+'mag'] - phot['ap2'], 0.0)
         wts = np.where(gids, np.power(phot['ap2er']**2 + \
               phot[filt+'err']**2,-1), 0.0)
         # 30 is used internall in photometry code as arbitrary zero-point
         zp = np.sum(diffs*wts)/np.sum(wts) + 30
         ezp = np.sqrt(1.0/np.sum(wts))
         self.log('Determined zero-point to be {} +/- {}'.format(
            zp,ezp))
         fts = fits.open(fil)
         fts[0].header['ZP'] = zp
         fts[0].header['EZP'] = ezp
         fts[0].writeto(fil, overwrite=True)
         self.initialPhot.append(fil)

      return

   def update_db(self):
      '''For all images that are fully reduced, calibrated, and template-
      subtracted, update photometry of the transient to the database.'''
      return

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
      while not self.stopped:
         print("Checking for new files")
         files = self.getNewFiles()
         for fil in files:
            time.sleep(wait_for_write)
            self.addFile(fil)

         self.BiasLinShutCorr()
         self.FlatCorr()

         self.identify()
         self.solve_wcs()
         self.photometry()
         #self.update_db()

         time.sleep(poll_interval)

      self.log("Pipeline stopped normally at {}".format(
         time.strftime('%Y/%m/%d %H:%M:%S')))
      return



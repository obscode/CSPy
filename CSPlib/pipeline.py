'''This module contains a pipeline class that does all the organizational
work of classifying images types, do the calibrations, and watching for
new files.'''

from astropy.io import fits
from . import ccdred
from . import headers
import os
from glob import glob
import time
import signal

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

      if not os.path.isdir(datadir):
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

      # Cache information so we don't have to open/close FITS files too often
      self.headerData = {}

      if workdir is None:
         self.workdir = self.datadir
      else:
         if not os.path.isdir(workdir):
            try:
               os.makedirs(workdir)
            except:
               raise OSError(
               "Cannot create workdir {}. Permission problem? Aborting".format(
                  workdir))
         self.workdir = workdir


      try:
         self.logfile = open(os.path.join(workdir, "pipeline.log"), 'w')
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
      if not os.path.isfile(filename):
         self.log("File {} not found. Did it disappear?".format(filename))
         return

      # Update header
      fout = os.path.join(self.workdir, os.path.basename(filename))
      fts = headers.update_header(filename, fout)

      fil = os.path.basename(filename)
      self.headerData[fil] = {}
      for h in ['OBJECT','OBSTYPE','FILTER','EXPTIME','OPAMP']:
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
      flist = glob(os.path.join(self.datadir, "{}*{}".format(
         self.prefix, self.suffix)))

      new = [f for f in flist if f not in self.rawfiles]

      return new

   def getHeaderData(self, fil, key):
      f = os.path.basename(fil)
      f = 'c'+f[1:]
      return self.headerData[f][key]

   def getFileName(self, fil, prefix):
      '''Add a prefix to the filename in the work folder.'''
      fil = os.path.basename(fil)
      return os.path.join(self.workdir, prefix+fil[1:])

   def makeBias(self):
      '''Make BIAS frame from the data, or retrieve from other sources.'''
      # Can we make a bias frame?
      if len(self.files['zero']) :
         self.log("Found {} bias frames, building an average...".format(
            len(self.files['zero'])))
         bfile = os.path.join(self.workdir, 'Zero{}'.format(self.suffix))
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
            self.biasFrame = fits.open(os.path.join(self.workdir, 
               'Zero{}'.format(self.suffix)))
         else:
            cfile = os.path.join(calibrations_folder, "CAL", "Zero{}".format(
               self.suffix))
            self.biasFrame = fts.open(cfile)
            bfile = os.path.join(self.workdir, "Zero{}".format(self.suffix))
            self.biasFrame.writeto(bfile)
            self.log("Retrieved backup BIAS frame from {}".format(cfile))

   def makeFlats(self):
      '''Make flat Frames from the data or retrieve from backup sources.'''

      for filt in filtlist:
         if len(self.files['sflat'][filt]) > 3:
            self.log("Found {} {}-band sky flats, bias and flux correcting..."\
                  .format(len(self.files['sflat'][filt]), filt))
            fname = os.path.join(self.workdir, "SFlat{}{}".format(filt,
               self.suffix))
            files = [self.getFileName(f,'b') for f in self.files['sflat'][filt]]
            self.flatFrame[filt] = ccdred.makeFlatFrame(files, outfile=fname)
            self.log("Flat field saved to {}".format(fname))
         else:
            cmd = "rclone copy CSP:Swope/Calibrations/latest/SFlat{}{} {}".\
                  format(filt,self.suffix,self.workdir)
            ret = os.system(cmd)
            if ret == 0:
               self.log("Retrieved Flat frame from latest reductions")
               self.flatFrame[filt] = fits.open(os.path.join(self.workdir, 
                  'SFlat{}{}'.format(filt,self.suffix)))
            else:
               cfile = os.path.join(calibrations_folder, "CAL", 
                     "SFlat{}{}".format(
                  filt, self.suffix))
               self.flatFrame[filt] = fits.open(cfile)
               self.flatFrame[filt].writeto(os.path.join(self.workdir,
                  "SFlat{}{}".format(filt,self.suffix)))
               self.log("Retrieved backup FLAT frame from {}".format(cfile))

   def BiasLinShutCorr(self):
      '''Do bias, linearity, and shutter corrections to all files except bias 
      frames.'''
      if self.biasFrame is None:
         self.log('Abort due to lack of bias frame')
         raise RuntimeError("Error:  can't proceed without a bias frame!")
      todo = []
      for f in self.rawfiles:
         base = os.path.basename(f)
         wfile = self.getFileName(f, 'c')
         bfile = self.getFileName(f, 'b')
         if wfile not in self.files['zero'] and bfile not in self.bfiles:
            todo.append(wfile)

      for f in todo:
         fts = ccdred.biasCorrect(f, overscan=True, frame=self.biasFrame)
         # Get the correct shutter file
         opamp = self.getHeaderData(f,'OPAMP')
         if opamp not in self.shutterFrames:
            shfile = os.path.join(calibrations_folder, 'CAL',
                  "SH{}.fits".format(opamp))
            self.shutterFrames[opamp] = fits.open(shfile)
         fts = ccdred.LinearityCorrect(fts)
         fts = ccdred.ShutterCorrect(fts, frame=self.shutterFrames[opamp])
         bfile = self.getFileName(f, 'b')
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
            bfile = self.getFileName(f, 'b')
            ffile = self.getFileName(f, 'f')
            if bfile in self.bfiles and ffile not in self.ffiles:  
               todo.append(bfile)

      for f in todo:
         filt = self.getHeaderData(f,'FILTER')
         bfile = self.getFileName(f, 'b')
         ffile = self.getFileName(f, 'f')
         if filt not in self.flatFrame:
            raise RuntimeError("No flat for filter {}. Abort!".format(filt))
         self.log("Flat field correcting {} --> {}".format(bfile,ffile))
         fts = ccdred.flatCorrect(bfile, self.flatFrame[filt],
               outfile=ffile)
         self.ffiles.append(ffile)

   def initialize(self):
      '''Make a first run through the data and see if we have what we need
      to get going. We can always fall back on generic calibrations if
      needed.'''

      self.log("Start pipeline at {}".format(time.strftime('%Y/%m/%d %H:%M:%S')))
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

         time.sleep(poll_interval)

      self.log("Pipeline stopped normally at {}".format(
         time.strftime('%Y/%m/%d %H:%M:%S')))
      return



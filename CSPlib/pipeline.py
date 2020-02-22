'''This module contains a pipeline class that does all the organizational
work of classifying images types, do the calibrations, and watching for
new files.'''

from astropy.io import fits
from . import ccdred
from . import headers
import os
from glob import glob

filtlist = ['u','g','r','i','B','V']
calibrations_folder = '/csp21/csp2/software/SWONC'

class Pipeline:

   def __init__(self, datadir, workdir=None, prefix='ccd', suffix='.fits',
         poll_interval=5):
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

   def makeBias(self):
      '''Make BIAS frame from the data, or retrieve from other sources.'''
      # Can we make a bias frame?
      if len(self.files['zero']) :
         self.log("Found {} bias frames, building an average...".format(
            len(self.files['zero'])))
         bfile = os.path.join(self.workdir, 'Zero{}'.format(self.suffix))
         self.biasFrame = ccdred.makeBias(self.files['zero'], 
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
            cfile = os.path.join(calibrations_folder, "Zero{}".format(
               self.suffix))
            self.biasFrame = fts.open(cfile)
            bfile = os.path.join(self.workdir, "Zero{}".format(self.suffix))
            self.biasFrame.writeto(bfile)
            self.log("Retrieved backup BIAS frame from {}".format(cfile))

   def makeFlats(self):
      '''Make flat Frames from the data or retrieve from backup sources.'''
      for filt in filtlist:
         if len(self.files['sflat'][filt]) > 3:
            self.log("Found {} {}-band sky flats, building an average..."\
                  .format(len(self.files['sflat'][filt]), filt))
            fname = os.path.join(self.workdir, "SFlat{}{}".format(filt,
               self.suffix))
            self.flatFrame[filt] = ccdred.makeFlatFrame(
                  self.files['sflat'][filt], outfile=fname)
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
               cfile = os.path.join(calibrations_folder, "SFlat{}{}".format(
                  filt, self.suffix))
               self.flatFrame[filt] = fits.open(cfile)
               self.flatFrame[filt].writeto(os.path.join(self.workdir,
                  "SFlat{}{}".format(filt,self.suffix)))
               self.log("Retrieved backup FLAT frame from {}".format(cfile))

   def initialize(self):
      '''Make a first run through the data and see if we have what we need
      to get going.'''

      files = self.getNewFiles()
      for fil in files:
         self.addFile(fil)

      self.makeBias()

      self.makeFlats()







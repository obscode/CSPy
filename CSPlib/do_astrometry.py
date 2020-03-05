'''A wrapper module to help a bit with solve-field.  Because most FITS
files from telescopes have RA/DEC headers, and perhaps pixel scale info,
we can get these and pass them to solve-field to get a good first
guess.  This, of course, assumes the FITS headers are reliable!.'''

import sys,os,re
import argparse
import subprocess

try:
   from astropy.io import fits
   have_pyfits = True
except:
   have_pyfits = False


# header keywords that might hold info we want
scale_keys = ['SCALE','PIXELSIZE','PIXELSCALE']
ra_keys = ['RA','RA-D','RA-OBS']
dec_keys = ['DEC','DEC-D','DEC-OBS']


def do_astrometry(files, trim=None, replace=False, dir='/usr/local/astromery',
      other=[], verbose=False):

   bindir=os.path.join(dir, 'bin')

   for fil in files:
      tmpfile = None
      sf_args = other
      if trim is not None:
         if not have_pyfits:
            if verbose: 
               print("Warning:  could not load fits module.  Can't trim.")
            filename = fil
         else:
            sec = trim
            res = re.search(r'\[(\*|\d+):(\*|\d+),(\*|\d+):(\*|\d+)\]', sec)
            if res is None:
               if verbose:
                  print("Error:  could not parse the section {}".format(sec))
                  print("        Using the full image")
               filename = fil
            else:
               ids = res.groups()
               ss = []
               for id in ids:
                  if id == '*':  
                     ss.append(None)
                  else:
                     ss.append(int(id)-1)
               sl = (slice(ss[2],ss[3]), slice(ss[0],ss[1]))
      
               # Create a temporary trimmed FITS file
               f = fits.open(fil)
               newdata = f[0].data[sl]
               tmpfile = fil + "_temp.fits"
               fits.writeto(tmpfile, newdata, f[0].header)
               f.close()
               if '-o' not in sf_args and '--out' not in sf_args:
                  sf_args += ['-o','.'.join(fil.split('.')[:-1])]
               filename = tmpfile
      else:
         filename = fil
      
      if have_pyfits:
         f = fits.open(filename)
         head = f[0].header
         scale = None
         ra = None
         dec = None
         for key in scale_keys:
            if key in head:
               scale = head[key]
               break
         for key in ra_keys:
            if key in head:
               ra = head[key]
               break
         for key in dec_keys:
            if key in head:
               dec = head[key]
               break
         
      if scale is not None:
         sf_args += ['-L',str(scale*0.95), '-H', str(scale*1.05), 
               '-u','arcsecperpix']
      elif ('-L' in sf_args and '-H' in sf_args) or \
           ('--scale-high' in sf_args and '--scale-low' in sf_args):
         if '-u' not in sf_args:
            sf_args += ['-u','arcsecperpix']
      else:
         if verbose:
            print("Warning:  couldn't find pixel scale in FITS header. You "
                  "might want to use --scale-high and --scale-low arguments")
      
      # put a default radius
      if '-5' not in sf_args and '--radius' not in sf_args:
         sf_args += ['--radius','1']
      
      if ra is None and ('-3' not in sf_args and '--ra' not in sf_args):
         if verbose:
            print("Warning:  couldn't find RA in header.  You might want to "
                  "use the --ra argument")
      else:
         sf_args += ['--ra',str(ra)]
      
      if dec is None and ('-4' not in sf_args and '--dec' not in sf_args):
         if verbose:
            print("Warning:  couldn't find DEC in header.  You might want to "
                  "use the --dec argument")
      else:
         sf_args += ['--dec',str(dec)]
      
      sf_args.insert(0,filename)
      
      e = os.path.join(bindir,'solve-field')
      if verbose:
         print("Running ",e+" "+' '.join(sf_args))
      
      # run the command
      # ret = os.system(e+" "+' '.join(sf_args))
      ret = subprocess.run([e] + sf_args, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
      logfile = '.'.join(fil.split('.')[:-1]) + ".log"
      with open(logfile, 'w') as fout:
         lines = ret.stdout.split(b'\n')
         for line in lines:
            fout.write(str(line)+"\n")
      if ret.returncode != 0:
         if verbose:
            print('solve-field failed for {}. Check {}.log'.format(fil,logfile))
         # Clean up
         if tmpfile is not None:
            os.unlink(tmpfile)
         return None
      
      # Clean up
      if tmpfile is not None:
         os.unlink(tmpfile)
      
      newfile = '.'.join(fil.split('.')[:-1])+'.new'
      if replace:
         os.system('mv {} {}'.format(newfile, fil))
         if have_pyfits:
            return fits.open(fil)
         else:
            return fil
      else:
         if have_pyfits:
            return fits.open(newfile)
         else:
            return newfile

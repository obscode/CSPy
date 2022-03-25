'''Module for downloading PanStarrs data (image cutouts and catalogs).'''

import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from astropy.wcs import WCS
import requests
try:
   import reproject
except:
   reproject = None

# Scale of PS images
PSscale = 0.25/3600   # in degrees/pixel

def getRemoteFITS(url, tries=5):
   '''Get remote FITS file and retry [tries] times if timeout.'''
   ntry = 0
   while ntry < tries:
      try:
         fts = fits.open(url)
         return fts
      except:
         print("******** timeout, retrying *********")
         ntry += 1

def getImages(ra, dec, size=240, filt='g', verbose=False):
   '''Query the PS data server to get a list of images for the given 
      coordinates, size and filter. We check the corners to see if
      we need to have more than one filename.

      Args:
         ra,dec (float):  RA/DEC in decimal degrees
         size (int):  Size of cutout in pixels
         filt (str): the filter you want
         verbose(bool): give extra info

      Returns:
         list of filenames
   '''
   base = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
   templ = "{}?ra={}&dec={}&size={}&format=fits&filters={}"
   baseurl = templ.format(base, ra, dec, size, filt)
   if verbose: print("About to query: " + baseurl)
   table = Table.read(baseurl, format='ascii')
   if len(table) == 0: return None
   filenames = [table[0]['filename']]
   for i in [-1,1]:
      for j in [-1,1]:
         url = templ.format(base, ra+i*size/2*PSscale, dec+j*size/2*PSscale,
               size,filt)
         table = Table.read(url, format='ascii')
         if table[0]['filename'] not in filenames:
            filenames.append(table[0]['filename'])
   return filenames


def geturls(ra, dec, size=240, filt='g'):
   '''Get the proper URL for PS images.

   Args:
      ra,dec(float):  coordinates in decimal degrees
      size (int):  size of image in pixels
      filt(str): which filter

   Returns:
      list of URLs.'''

   flist = getImages(ra, dec, size, filt)
   if flist is None:
      # No data found
      return None
   base = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
   templ = "{}?ra={}&dec={}&size={}&format=fits&red={}"
   urls = [templ.format(base, ra, dec, size, f) for f in flist]
   return urls

def getFITS(ra, dec, size, filters, mosaic=False):
   '''Retrieve the FITS files from PanSTARRS server, centered on ra,dec
   and with given size.

   Args:
      ra (float):  RA in degrees
      dec (float):  DEC in degrees
      size (float):  size of FOV in degrees
      filters (str):  filters to get:  e.g gri
      mosaic(bool): If more than one PS images is needed to tile the field,
                    do we mosaic them? Requires reproject module if True

   Returns:
      list of FITS instances
   '''
   isize = int(size/PSscale)
   # If the size is big enough, we can hit the limits of PS fields, in which
   # case we get data back with a bunch of NaNs. So we need to check if we
   # need multiple queries.
   if mosaic and reproject is None:
      raise ValueError("To use mosaic, you need to install reproject")
   filters = list(filters)
   ret = []
   for filt in filters:
      urls = geturls(ra, dec, isize, filt)
      if urls is None:
         # no data
         return None
      if len(urls) > 1 and mosaic:
         from reproject import reproject_interp
         baseurl = urls[0]
         #basefts = fits.open(baseurl)
         basefts = getRemoteFITS(baseurl)
         for url in urls[1:]:
            #ft = fits.open(url)
            ft = getRemoteFITS(url)
            arr,foot = reproject_interp(ft[0], basefts[0].header)
            intersect = np.isnan(basefts[0].data)*np.greater(foot, 0)
            basefts[0].data = np.where(intersect, arr, basefts[0].data)
         ret.append(basefts)
      else:
         #ret.append(fits.open(urls[0]))
         ret.append(getRemoteFITS(urls[0]))

   return ret


def ps1cone(ra,dec,radius,table="mean",release="dr1",format="csv",columns=None,
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
    **kw):
   """Do a cone search of the PS1 catalog
   
   Args:
      ra (float): (degrees) J2000 Right Ascension
      dec (float): (degrees) J2000 Declination
      radius (float): (degrees) Search radius (<= 0.5 degrees)
      table (string): mean, stack, or detection
      release (string): dr1 or dr2
      format: csv, votable, json
      columns: list of column names to include (None means use defaults)
      baseurl: base URL for the request
      verbose: print info about request
      **kw: other parameters (e.g., 'nDetections.min':2)
   """
   
   data = kw.copy()
   data['ra'] = ra
   data['dec'] = dec
   data['radius'] = radius
   return ps1search(table=table,release=release,format=format,columns=columns,
      baseurl=baseurl, verbose=verbose, **data)

def ps1search(table="mean",release="dr1",format="csv",columns=None,
   baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
   **kw):
   """Do a general search of the PS1 catalog (possibly without ra/dec/radius)
   
   Args:
      table (string): mean, stack, or detection
      release (string): dr1 or dr2
      format: csv, votable, json
      columns: list of column names to include (None means use defaults)
      baseurl: base URL for the request
      verbose: print info about request
      **kw: other parameters (e.g., 'nDetections.min':2). Note this is required!
   """
   
   data = kw.copy()
   if not data:
      raise ValueError("You must specify some parameters for search")
   checklegal(table,release)
   if format not in ("csv","votable","json"):
      raise ValueError("Bad value for format")
   url = "{baseurl}/{release}/{table}.{format}".format(**locals())
   if columns:
      # check that column values are legal
      # create a dictionary to speed this up
      dcols = {}
      for col in ps1metadata(table,release)['name']:
         dcols[col.lower()] = 1
      badcols = []
      for col in columns:
         if col.lower().strip() not in dcols:
            badcols.append(col)
      if badcols:
         raise ValueError('Some columns not found in table: {}'\
               .format(', '.join(badcols)))
      # two different ways to specify a list of column values in the API
      # data['columns'] = columns
      data['columns'] = '[{}]'.format(','.join(columns))

# either get or post works
#    r = requests.post(url, data=data)
   r = requests.get(url, params=data)

   if verbose:
      print(r.url)
   r.raise_for_status()
   if format == "json":
      return r.json()
   else:
      return r.text


def checklegal(table,release):
   """Checks if this combination of table and release is acceptable
   
   Args:
      table (str): table to check
      release (str): Data release (dr1, dr2)

   Returns:
      None
      
   Effects:
      Raises a VelueError exception if there is problem
   """

   releaselist = ("dr1", "dr2")
   if release not in ("dr1","dr2"):
      raise ValueError("Bad value for release (must be one of {})".\
            format(', '.join(releaselist)))
   if release=="dr1":
      tablelist = ("mean", "stack")
   else:
      tablelist = ("mean", "stack", "detection")
   if table not in tablelist:
      raise ValueError("Bad value for table (for {} must be one of {})".\
            format(release, ", ".join(tablelist)))

def ps1metadata(table="mean",release="dr1",
       baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
   """Return metadata for the specified catalog and table

   Args:
      table (string): mean, stack, or detection
      release (string): dr1 or dr2
      baseurl: base URL for the request

   Returns:
      An astropy table with columns name, type, description
   """

   checklegal(table,release)
   url = "{baseurl}/{release}/{table}/metadata".format(**locals())
   r = requests.get(url)
   r.raise_for_status()
   v = r.json()
   # convert to astropy table
   tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
      names=('name','type','description'))
   return tab

def mastQuery(request):
   """Perform a MAST query.
   
   Args:
      request (dictionary): The MAST request json object
   
   Returns:
      head,content where head is the response HTTP headers, and 
      content is the returned data
   """
   
   server='mast.stsci.edu'
   # Grab Python Version 
   version = ".".join(map(str, sys.version_info[:3]))
   # Create Http Header Variables
   headers = {"Content-type": "application/x-www-form-urlencoded",
              "Accept": "text/plain",
              "User-agent":"python-requests/"+version}

   # Encoding the request as a json string
   requestString = json.dumps(request)
   requestString = urlencode(requestString)
   
   # opening the https connection
   conn = httplib.HTTPSConnection(server)
   # Making the query
   conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

   # Getting the response
   resp = conn.getresponse()
   head = resp.getheaders()
   content = resp.read().decode('utf-8')

   # Close the https connection
   conn.close()

   return head,content

def resolve(name):
   """Get the RA and Dec for an object using the MAST name resolver

   Args:
      name (str): Name of object

   Returns:
      (RA, Dec) tuple with position"""

   resolverRequest = {'service':'Mast.Name.Lookup',
                      'params':{'input':name,
                                'format':'json'
                               },
                      }
   headers,resolvedObjectString = mastQuery(resolverRequest)
   resolvedObject = json.loads(resolvedObjectString)
   # The resolver returns a variety of information about the resolved object, 
   # however for our purposes all we need are the RA and Dec
   try:
      objRa = resolvedObject['resolvedCoordinate'][0]['ra']
      objDec = resolvedObject['resolvedCoordinate'][0]['decl']
   except IndexError as e:
      raise ValueError("Unknown object '{}'".format(name))
   return (objRa, objDec)

def getStarCat(ra, dec, radius):
   '''Get a list of PS stars plus their photometry.
   
   Args:
      ra (float): RA in decimal degrees.
      dec (float): DEC in decimal degrees
   
   Returns:
      astropy.table with catalog data.
   '''
   columns = ['objID','raMean','decMean']
   for filt in ['g','r','i']:
      for col in ['MeanPSFMag','MeanPSFMagErr']:
         columns.append(filt+col)
   contraints = {'nDetections.gt':1,
                 'gMeanPSFMag.gt':0,
                 'gMeanPSFMagErr.gt':0,
                 'rMeanPSFMag.gt':0,
                 'rMeanPSFMagErr.gt':0,
                 'iMeanPSFMag.gt':0,
                 'iMeanPSFMagErr.gt':0}
   results = ps1cone(ra, dec, radius, release='dr2', columns=columns,
         table='mean', **contraints)
   if results == '':
      return None
   tab = ascii.read(results)
   # make some nice column names
   tab.rename_column('raMean','RA')
   tab.rename_column('decMean','DEC')
   for filt in ['g','r','i']:
      tab.rename_column(filt+'MeanPSFMag', filt+'mag')
      tab.rename_column(filt+'MeanPSFMagErr', filt+'err')
   return tab


'''A module with convenience functions for connecting to the CSP database.'''
import getpass
from astropy.table import Table
from astropy.time import Time
import pymysql
import os
from numpy import argsort,array
from datetime import date
from .config import getconfig
import re

cfg = getconfig()

dbs = {'SBS': {
         'host':'sql.obs.carnegiescience.edu',
         'user':'CSP',
         'db':'CSP'},
       'SBSpub':{
         'host':'sql.obs.carnegiescience.edu',
         'user':'CSP',
         'db':'CSPpub'},
       'LCO': {
          'host':'csp2.lco.cl',
          'user':'cburns',
          'db':'Phot'},
       'SSH': {
          'host':'localhost',
          'user':'cburns',
          'db':'Phot'},
       'POISE':{
          'host':'sql.obs.carnegiescience.edu',
          'user':'cburns',
          'db':'POISE'},
       }
              
# Campaign strings to numbers
camppat = re.compile(r'([0-9]{4,4})([AB])')
def camp2num(camp):
   res = camppat.search(camp)
   if not res:
      raise ValueError("Unrecognized campaign string")
   year,code = res.groups()
   return 19 + (int(year)-2023)*2 + ['A','B'].index(code)

def num2camp(num):
   code = ['B','A'][num % 2]
   return str(2023 + (num-19)//2)+code

   

if 'CSPpasswd' in os.environ:
   passwd = os.environ['CSPpasswd']
else:
   passwd = cfg.remote.CSPpasswd

if 'CSPdb' in os.environ:
   default_db = os.environ['CSPdb']
else:
   default_db = cfg.remote.CSPdb

def getConnection(db=default_db):
   global passwd, dbs
   if db not in dbs:
      raise ValueError("db must be either SBS or LCO")
   if passwd is None:
      resp = getpass.getpass(
            prompt="SQL passwd:")
   else:
      resp = passwd
   d = pymysql.connect(passwd=resp, **dbs[db])
   passwd = resp
   return d

def getPhotometricNights(SN, db=default_db):
   '''Given a SN name, search the database for nights that were photometric.
   
   Args:
      SN (str):  Name of SN
      
   Returns
      nights:  list of datetime() dates
   '''
   db = getConnection(db=db)
   c = db.cursor()
   c.execute('''select MAGINS.night from MAGINS left join MAGFIT1
                on (MAGINS.night=MAGFIT1.night) 
                where MAGINS.field=%s and MAGFIT1.comm="ok" 
                group by MAGINS.night''', (SN,))
   nights = c.fetchall()
   db.close()
   return [item[0] for item in nights]

def getPhotometricZeroPoints(SN, filt, db=default_db):
   '''Given a supernova and filter, find all photometric nights and
   return their zero-points.

   Args:
      SN (str):  Supernova name
      filt (str):  filter
      
   Returns:
      table:  astropy.Table with columns 'night','zp','zper'. 
   '''
   db = getConnection(db=db)
   c = db.cursor()
   c.execute('''select MAGINS.night,MAGFIT1.zp,MAGFIT1.zper 
                FROM MAGINS left join MAGFIT1
                on (MAGINS.night=MAGFIT1.night and MAGINS.filt=MAGFIT1.filt) 
                where MAGINS.field=%s and MAGFIT1.comm="ok" and MAGINS.filt=%s
                GROUP by night
                ''', (SN,filt))
   data = c.fetchall()
   db.close()
   tab = Table(rows=data, names=['night', 'zp', 'zper'])
   return tab
   

def getStandardPhotometry(SN, filt, db=default_db):
   '''Given a SN name, retrieve the instrumental magnitudes, airmasses
   for standard stars on nights that were photometric.
   
   Args:
      SN (str):  Name of the SN
      filt (str):  Name of the filter
      
   Returns:
      table:  astropy.Table with columns 'OBJ','night','magins','emagins
              'airm','zp',zper'.
   '''
   nights = getPhotometricZeroPoints(SN, filt)
   db = getConnection(db=db)
   c = db.cursor()
   data = []
   for i in range(len(nights)):
      night = nights[i]['night']
      c.execute('''SELECT field,night,ap7,ap7er,airm
                   FROM MAGINS
                   WHERE night=%s and filt=%s''', (night, filt))
      res = c.fetchall()
      for line in res:
         if line[0] in optstd['OBJ']:
            data.append(list(line) + [nights[i]['zp'], nights[i]['zper']])

   tab = Table(rows=data, names=['OBJ','night','magins','emagins','airm',
                                 'zp','zper'])
   for col in ['magins','emagins','zp','zper','airm']: 
      tab[col].info.format='%.3f'
   return tab

def getDBName(name, db=default_db):
   '''Given a name, return the DB primary name from the database. Return -1 if
   name not found and -2 if connection fails.
   
   Args:
      name(str):  Name of the object
      db(str):  Name of the database
      
   Returns:
      IAUName:  the IAU name
   '''
   try:
      db = getConnection(db)
   except:
      return -2
   c = db.cursor()
   res = c.execute(
         "select SN from SNList where (SN=%s or NAME_IAU=%s or NAME_CSP=%s "\
         "or NAME_PSN=%s)", (name, name, name, name))
   if res < 1:
      return -1
   return c.fetchall()[0]

def getAllNames(name, db=default_db):
   '''Given a name, return All names from the database. Return -1 if
   name not found and -2 if connection fails.
   
   Args:
      name(str):  Name of the object
      db(str):  Name of the database
      
   Returns:
      dict:  keyed by 'DB','CSP','IAU','PSN'
   '''
   try:
      db = getConnection(db)
   except:
      return -2
   c = db.cursor()
   res = c.execute(
         "select SN,NAME_CSP,NAME_IAU,NAME_PSN from SNList where (SN=%s "\
         "or NAME_IAU=%s or NAME_CSP=%s or NAME_PSN=%s)", 
         (name, name, name, name))

   if res < 1:
      return -1
   res = c.fetchall()[0]
   return dict(DB=res[0], CSP=res[1], IAU=res[2], PSN=res[3])


def getNameCoords(name, db=default_db):
   '''Given a name, return the coordinates or -1 if not found or -2 if
   connection fails.
   
   Args:
      name(str):  Name of the object
      db(str):  Name of the database
      
   Returns:
      (RA,DEC):  tuple of floats, both in degrees
   '''

   try:
      db = getConnection(db)
   except:
      return -2
   c = db.cursor()
   c.execute('''SELECT RA*15,DE from SNList where SN=%s''', name)
   l = c.fetchall()
   if len(l) == 0:
      return -1

   return(l[0])

def getCoordsName(ra, dec, db=default_db, tol=0.125):
   '''Given coordinates, find a name within tol degrees. Return -1 if nothing
   found, -2 if database can't be reached.
   
   Args:
      ra(float):  RA in decimal degrees
      dec(float):  DEC in decimal degrees
      db(str):  database code
      tol(float):  maximum distance to consier match in degrees.'''
   try:
      db = getConnection(db)
   except:
      return -2
   ra = float(ra)
   dec = float(dec)

   c = db.cursor()
   c.execute("SELECT SN,RA*15,DE,SQRT(POW((RA*15-%s)*COS(%s/180*3.14159),2) + "
             "POW((DE-%s),2)) as dist FROM "
             "SNList having dist < %s ORDER BY dist",
             (ra,dec,dec,tol))
   l = c.fetchall()
   if len(l) == 0:
      return -1
   # If More than one, use closest (first)
   return l[0]

def getLSMags(SN, filt, db=default_db):
   '''Retrieve the magnitudes of the local sequence stars.
   
   Args:
      SN(str):  Name of the SN
      filt(str):  Name of the filter
      db(sr):   Database to query
   
   Returns:
      tab(astropy.table):  Table of object ID, mag, and emag'''

   try:
      db = getConnection(db)
   except:
      return -2
   c = db.cursor()
   res = c.execute("SELECT obj,mag,err,ncal,mag2,err2 FROM MAGLSEQ "\
                   "where field=%s and filt=%s", (SN,filt))
   if res < 1:
      return -1
   rows = c.fetchall()
   tab = Table(rows=rows, names=['objID','mag','err','ncal','mag2','err2'])
   return(tab)



def getLSCoords(SN, db=default_db):
   '''Retrieve the coordinates of the local sequence stars.
   
   Args:
      SN(str):  Name of the SN
      db(sr):   Database to query
   
   Returns:
      tab(astropy.table):  Table of object ID, RA, and DEC (in degrees)'''

   try:
      db = getConnection(db)
   except:
      return -2
   c = db.cursor()
   res = c.execute("SELECT OBJ,RA*15,DE FROM Coords where SN=%s", (SN,))
   if res < 1:
      return -1
   rows = c.fetchall()
   tab = Table(rows=rows, names=['objID','RA','DEC'])
   return(tab)


def updateSNPhot(SN, JD, filt, fits, mag, emag, db=default_db):
   '''Update the SN photometry in the database.

   Args:
      SN(str):  Name of the SN object
      JD(float):  Julian day
      filt(str):  Name of the filter
      fits(str):  FITS file name.
      mag(float):  magnitude
      emag(float):  error in magnitude
      db(str):  name of the database

   Returns:
      None

   Effects:
      Photometry is updated in the database.
   '''

   try:
      db = getConnection(db)
   except:
      return -2
   JD = float(JD)
   mag = float(mag)
   emag = float(emag)
   t = Time(JD, format='jd').datetime.date()
   c = db.cursor()
   
   #check to see if entry exists:
   n = c.execute('select night,fits from MAGSN where fits=%s and night=%s', 
         (fits,t))
   if n > 0:
      c.execute('delete from MAGSN where fits=%s and night=%s', (fits,t))

   c.execute('INSERT INTO MAGSN (night,field,obj,filt,fits,mag,err,jd) '\
             'VALUES (%s,%s,%s,%s,%s,%s,%s,%s)', 
             (t, SN, 0, filt, fits, mag, emag, JD))


   db.close()


def getSpectraList(SN, db=default_db):
   '''Get a list of spectra that are available on the database for the object
   SN
   
   Args:
      SN(str):  Supernova name
      db(str):  The database to access.

   Returns:
      spectra(list):  list of spectra. Empty list denotes no spectra found.
   '''

   try:
      db = getConnection(db)
   except:
      return []

   c = db.cursor()

   N = c.execute('''SELECT * from SP_INFO where SN=%s''', (SN,))
   if N == 0:
      return []
   data = c.fetchall()
   return [[d[1],d[2],d[3],d[4],d[5]] for d in data]

def getSpectrum(filename, db=default_db):
   '''Get a single spectrum given a filename.

   Args:
      filename(str):  Name of the FITS file (found using getSpectraList)
      db(str):  the databse to query

   Returns:
      lambda,flux:  2-tuple of numpy arrays. The wavelenth in Angstroms and
                    flux in erg/s/cm**2/Angs
   '''

   try:
      db = getConnection(db)
   except:
      return None

   c = db.cursor()
   N = c.execute('''SELECT LAMBDA,FLUX from SPECTRA where FILE=%s''', 
         (filename,))
   if N == 0: return None
   data = c.fetchall()
   return [array([d[0] for d in data]),array([d[1] for d in data])]

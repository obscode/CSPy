'''A module with convenience functions for connecting to the CSP database.'''
import getpass
from astropy.table import Table
import pymysql
import os

if 'CSPpasswd' in os.environ:
   passwd = os.environ['CSPpasswd']
else:
   passwd = None

def getConnection():
   global passwd
   if passwd is None:
      resp = getpass.getpass(
            prompt="SQL passwd for CSP@sql.obs.carnegiescience.edu:")
   else:
      resp = passwd
   db = pymysql.connect(host='sql.obs.carnegiescience.edu',
      user='CSP', passwd=resp, db='CSP')
   passwd = resp
   return db

def getPhotometricNights(SN):
   '''Given a SN name, search the database for nights that were photometric.
   
   Args:
      SN (str):  Name of SN
      
   Returns
      nights:  list of datetime() dates
   '''
   db = getConnection()
   c = db.cursor()
   c.execute('''select MAGINS.night from MAGINS left join MAGFIT1
                on (MAGINS.night=MAGFIT1.night) 
                where MAGINS.field=%s and MAGFIT1.comm="ok" 
                group by MAGINS.night''', (SN,))
   nights = c.fetchall()
   db.close()
   return [item[0] for item in nights]

def getPhotometricZeroPoints(SN, filt):
   '''Given a supernova and filter, find all photometric nights and
   return their zero-points.

   Args:
      SN (str):  Supernova name
      filt (str):  filter
      
   Returns:
      table:  astropy.Table with columns 'night','zp','zper'. 
   '''
   db = getConnection()
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
   

def getStandardPhotometry(SN, filt):
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
   db = getConnection()
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



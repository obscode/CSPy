'''A python module that deals with photometric transformations between
CSP natural and standard systems. Mostly this deals with color-terms.'''
from numpy import *
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
basedir = os.path.join(os.path.dirname(os.path.realpath(os.path.dirname(__file__))),'data')

optstd = ascii.read(os.path.join(basedir, 'opt.std.cat'), 
      fill_values=('INDEF',-99),
      names=['OBJ','RAh','RAm','RAs','DECd','DECm','DECs','V','BV','UB',
             'VR','RI','VI','eV','eBV','eUB','eVR','eRI','eVI',
             'r','ug','gr','ri','iz','er','eug','egr','eri','eiz'])
nirstd = ascii.read(os.path.join(basedir, 'nir.std.cat'),
      fill_values=('INDEF', -99),
      names=['OBJ','index','RA','DEC','epoch','Y','eY','J','eJ','H','eH','K',
         'eK','Ks','eKs'])
cs = SkyCoord(nirstd['RA'], nirstd['DEC'], unit=(u.hourangle, u.degree))
nirstd['RAd'] = cs.ra.value
nirstd['DECd'] = cs.dec.value

cts = {('SWO','DC'):{'u':0.046,
                     'g':-0.014,
                     'r':-0.016,
                     'i':-0.002,
                     'B':0.061,
                     'V':-0.058},
       ('SWO','NC'):{'u':0.030,
                     'g':-0.005,
                     'r':-0.001,
                     'i':0.021,
                     'B':0.091,
                     'V':-0.062},
       ('SWO','RC'):{'Y':0.00,
                     'J':0.016,
                     'H':-0.029},
       ('DUP','WI'):{'Y':-0.042,
                     'J':0.016,
                     'H':-0.029},
       ('DUP','RC'):{'Y':0.00,
                     'J':0.019,
                     'H':-0.039},
       ('BAA','FS'):{'Y':0.106,
                     'J':0.001,
                     'H':-0.040}
       }


ects = {('SWO','DC'):{'u':0.017,
                     'g':0.011,
                     'r':0.015,
                     'i':0.015,
                     'B':0.012,
                     'V':0.011},
       ('SWO','NC'):{'u':0.017,
                     'g':0.011,
                     'r':0.015,
                     'i':0.015,
                     'B':0.012,
                     'V':0.011},
       ('SWO','RC'):{'Y':0.00,
                     'J':0.00,
                     'H':-0.0},
       ('DUP','WI'):{'Y':0.0,
                     'J':0.0,
                     'H':0.0},
       ('DUP','RC'):{'Y':0.00,
                     'J':0.0,
                     'H':0.0},
       ('BAA','FS'):{'Y':0.0,
                     'J':0.0,
                     'H':0.0}
       }

colors = {'B':('B','V'),
          'V':('V','i'),
          'u':('u','g'),
          'g':('g','r'),
          'r':('r','i'),
          'i':('r','i')}

kX = {'u':0.51, 'g':0.19, 'r':0.11, 'i':0.07, 'B':0.24, 'V':0.14}


def stand2nat(up,gp,rp,ip,B,V, tel='SWO', ins='DC'):
   '''Convert from optical standard photometry to natural photometry.

   Args:
      up,gp,rp,ip,B,V (floats):  Standard magnitudes in ugri and BV
      tel (str):  telescope (SWO, DUP, or BAA)
      ins (str):  instrument (DC, NC, etc)

   Returns:
      (u,g,r,i,b,v):  the natural system magnitudes
   '''

   b = B - cts[(tel,ins)]['B']*(B - V)
   v = V - cts[(tel,ins)]['V']*(V - ip)
   u = up - cts[(tel,ins)]['u']*(up - gp)
   g = gp - cts[(tel,ins)]['g']*(gp - rp)
   r = rp - cts[(tel,ins)]['r']*(rp - ip)
   i = ip - cts[(tel,ins)]['i']*(rp - ip)
   return u,g,r,i,b,v

def NIRstand2nat(Y, J, H, tel='SWO', ins='RC'):
   '''Convert from NIR natural photometry to standard photometry.

   Args:
      Y,J,H (floats):  Standard  magnitudes in YJH
      tel (str):  telescope (SWO, DUP, or BAA)
      ins (str):  instrument (WI, RC, FS, etc)

   Returns:
      (y,j,h):  the natural system magnitudes
   '''
   y = Y - cts[(tel,ins)]['Y']*(J - H)
   j = J - cts[(tel,ins)]['J']*(J - H)
   h = H - cts[(tel,ins)]['H']*(J - H)
   return (y,j,h)

def nat2stand(u,g,r,i,b,v, tel='SWO', ins='DC'):
   '''Convert from natural photometry to standard photometry.

   Args:
      u,g,r,i,b,v (floats):  Natural  magnitudes in ugri and BV
      tel (str):  telescope (SWO, DUP, or BAA)
      ins (str):  instrument (DC, NC, etc)

   Returns:
      (up,gp,rp,ip,B,V):  the standard system magnitudes
   '''
   c1ri = 1 - cts[(tel,ins)]['r'] - cts[(tel,ins)]['i']
   rp = r + cts[(tel,ins)]['r']*(r - i)/c1ri
   gp = (g - cts[(tel,ins)]['g']*rp)/(1 - cts[(tel,ins)]['g'])
   up = (u - cts[(tel,ins)]['u']*gp)/(1 - cts[(tel,ins)]['u'])
   ip = i + cts[(tel,ins)]['i']*(r - i)/c1ri
   V = (v - cts[(tel,ins)]['V']*ip)/(1 - cts[(tel,ins)]['V'])
   B = (b - cts[(tel,ins)]['B']*V)/(1 - cts[(tel,ins)]['B'])
   return (up,gp,rp,ip,B,V)

def NIRnat2stand(y, j, h, tel='SWO', ins='DC'):
   '''Convert from NIR standard photometry to natural photometry.

   Args:
      y,j,h (floats):  natural  magnitudes in YJH
      tel (str):  telescope (SWO, DUP, or BAA)
      ins (str):  instrument (WI, RC, FS, etc)

   Returns:
      (Y,J,H):  the standard system magnitudes
   '''
   cY = cts[(tel,ins)]['Y']
   cJ = cts[(tel,ins)]['J']
   cH = cts[(tel,ins)]['H']
   JH = (j - h)/(1 - cJ + cH)
   Y = y + cY*JH
   J = j + cJ*JH
   H = h + cH*JH
   return Y,J,H
   
def getOptStandardMag(filt, names=None):
   '''Returns a table of standard magnitudes. If `names` is supplied, only
   return those names.

   Args: 
      filt (str):  Filter (one of ugriBV)
      names (list): list of stndards
      
   Returns:
      tab (astropy.table):  table with OBJ, mag, emag
      '''

   if names is not None:
      gids = array([name in names for name in optstd['OBJ']])
      tab = optstd[gids]
   else:
      tab = optstd[:]

   if filt == 'V':
      tab = tab['OBJ','V','eV']
   elif filt == 'r':
      tab = tab['OBJ','r','er']
   elif filt == 'B':
      tab = Table([tab['OBJ'], tab['BV']+tab['V'],
         sqrt(power(tab['eBV'],2)+power(tab['eV'],2))])
   elif filt == 'g':
      tab = Table([tab['OBJ'], tab['gr']+tab['r'],
         sqrt(power(tab['egr'],2)+power(tab['er'],2))])
   elif filt == 'i':
      tab = Table([tab['OBJ'], tab['r']-tab['ri'],
         sqrt(power(tab['eri'],2)+power(tab['er'],2))])
   elif filt == 'u':
      tab = Table([tab['OBJ'], tab['r']+tab['gr']+tab['ug'],
         sqrt(power(tab['egr'],2)+power(tab['eug'],2) + power(tab['er'],2))])
   tab.rename_column(tab.colnames[1], 'mag')
   tab.rename_column(tab.colnames[2], 'emag')
   return tab

def getOptNaturalMag(filt, names=None, tel='SWO', ins='DC'):
   '''Returns a table of natural magnitudes. If `names` is supplied, only
   return those names.

   Args: 
      filt (str):  Filter (one of ugriBV)
      names (list): list of stndards
      tel (str):  telescope code (SWO, DUP, BAA)
      ins (str):  instrument code (DC, NC, WI, etc)
      
   Returns:
      tab (astropy.table):  table with OBJ, mag, emag
      '''
   bands = ['u','g','r','i','B','V']
   objs = [getOptStandardMag(band, names)['OBJ'] for band in bands]
   mags = [getOptStandardMag(band, names)['mag'] for band in bands]
   emags = [getOptStandardMag(band, names)['emag'] for band in bands]

   res = stand2nat(*mags, tel=tel, ins=ins)
   idx = bands.index(filt)
   
   return Table([objs[idx], res[idx], emags[idx]],
         names=['OBJ','mag','emag'])

def getNIRStandardMag(filt, names=None):
   '''Returns a table of NIR standard magnitudes. If `names` is supplied, only
   return those names.

   Args: 
      filt (str):  Filter (one of YJHKKs)
      names (list): list of standards
      
   Returns:
      tab (astropy.table):  table with OBJ, mag, emag
      '''

   if names is not None:
      gids = array([name in names for name in nirstd['OBJ']])
      tab = nirstd[gids]
   else:
      tab = nirstd[:]

   tab = Table([tab['OBJ'], tab[filt], tab['e'+filt]])
   tab.rename_column(tab.colnames[1], 'mag')
   tab.rename_column(tab.colnames[2], 'emag')
   return tab

def getNIRNaturalMag(filt, names=None, tel='SWO', ins='RC'):
   '''Returns a table of natural NIR magnitudes. If `names` is supplied, only
   return those names.

   Args: 
      filt (str):  Filter (one of YJHKs)
      names (list): list of stndards
      tel (str):  telescope code (SWO, DUP, BAA)
      ins (str):  instrument code (DC, NC, WI, etc)
      
   Returns:
      tab (astropy.table):  table with OBJ, mag, emag
      '''
   bands = ['Y','J','H']
   objs = [getNIRStandardMag(band, names)['OBJ'] for band in bands]
   mags = [getNIRStandardMag(band, names)['mag'] for band in bands]
   emags = [getNIRStandardMag(band, names)['emag'] for band in bands]

   res = NIRstand2nat(*mags, tel=tel, ins=ins)
   idx = bands.index(filt)
   
   return Table([objs[idx], res[idx], emags[idx]],
         names=['OBJ','mag','emag'])

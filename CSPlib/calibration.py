'''A python module that deals with calibration and photometric transformations 
between CSP natural and standard systems. Mostly this deals with color-terms.

2021/03/12:  Added in color terms and transformations to go from
             panstarrs and skymapper mags to CSP mags

2021/04/02:  Renamed to calibration.py. Putting all zp-related stuff here too'''
from numpy import *
from astropy.io import ascii
from astropy.table import Table,join
from astropy.coordinates import SkyCoord
from scipy.interpolate import splev
from astropy import units as u
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import transforms
import re
import warnings
basedir = os.path.join(os.path.realpath(os.path.dirname(__file__)),'data')

optstd = ascii.read(os.path.join(basedir, 'opt.std.cat'), 
      fill_values=('INDEF',-99),
      names=['OBJ','RAh','RAm','RAs','DECd','DECm','DECs','V','BV','UB',
             'VR','RI','VI','eV','eBV','eUB','eVR','eRI','eVI',
             'r','ug','gr','ri','iz','er','eug','egr','eri','eiz'])
optcoord = ascii.read(os.path.join(basedir, 'STDS.txt'))
optstd = join(optstd,optcoord, keys='OBJ')
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

kX = {('SWO','DC'):{
         'u':0.504, 'g':0.193, 'r':0.103, 'i':0.06, 'B':0.244, 'V':0.141},
      ('SWO','NC'):{
         'u':0.507, 'g':0.186, 'r':0.090, 'i':0.051, 'B':0.231, 'V':0.136}
      }




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
      tab = tab['OBJ','RA','DEC','V','eV']
   elif filt == 'r':
      tab = tab['OBJ','RA','DEC','r','er']
   elif filt == 'B':
      tab = Table([tab['OBJ'],tab['RA'],tab['DEC'], tab['BV']+tab['V'],
         sqrt(power(tab['eBV'],2)+power(tab['eV'],2))])
   elif filt == 'g':
      tab = Table([tab['OBJ'],tab['RA'],tab['DEC'], tab['gr']+tab['r'],
         sqrt(power(tab['egr'],2)+power(tab['er'],2))])
   elif filt == 'i':
      tab = Table([tab['OBJ'],tab['RA'],tab['DEC'], tab['r']-tab['ri'],
         sqrt(power(tab['eri'],2)+power(tab['er'],2))])
   elif filt == 'u':
      tab = Table([tab['OBJ'],tab['RA'],tab['DEC'],tab['r']+tab['gr']+tab['ug'],
         sqrt(power(tab['egr'],2)+power(tab['eug'],2) + power(tab['er'],2))])
   tab.rename_column(tab.colnames[3], 'mag')
   tab.rename_column(tab.colnames[4], 'emag')
   return tab

def getOptStandardColor(f1, f2, names=None):
   '''Returns a table of standard colors. If `names` is supplied, only
   return those names.

   Args: 
      f1 (str):  Filter 1 of the color (f1-f2) (one of ugriBV)
      f2 (str):  Filter 2 of the color (f1-f2) (one of ugriBV)
      names (list): list of stndards
      
   Returns:
      tab (astropy.table):  table with OBJ, color, ecolor
      '''

   tab1 = getOptStandardMag(f1, names=names)
   tab2 = getOptStandardMag(f2, names=names)
   tab = join(tab1, tab2, keys='OBJ')
   tab['color'] = tab['mag_1'] - tab['mag_2']
   with warnings.catch_warnings():    # disable sqrt warnings
      warnings.simplefilter('ignore')
      tab['ecolor'] = sqrt(tab['emag_1']**2 + tab['emag_2']**2)
   tab = tab['OBJ','color','ecolor']
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
   tabs = [getOptStandardMag(band, names) for band in bands]
   mags = [t['mag'] for t in tabs]
   emags = [t['emag'] for t in tabs]
   objs = tabs[0]['OBJ']
   RAs = tabs[0]['RA']
   DECs = tabs[0]['DEC']

   res = stand2nat(*mags, tel=tel, ins=ins)
   idx = bands.index(filt)
   
   return Table([objs, RAs, DECs, res[idx], emags[idx]],
         names=['OBJ','RA','DEC','mag','emag'])

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

   tab = tab['OBJ','RAd','DECd',filt,'e'+filt]
   tab.rename_column('RAd','RA')
   tab.rename_column('DECd','DEC')
   tab.rename_column(tab.colnames[3], 'mag')
   tab.rename_column(tab.colnames[4], 'emag')
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
   tabs = [getNIRStandardMag(band, names) for band in bands]
   objs = tabs[0]['OBJ']
   RAs = tabs[0]['RA']
   DECs = tabs[0]['DEC']
   mags = [t['mag'] for t in tabs]
   emags = [t['emag'] for t in tabs]

   res = NIRstand2nat(*mags, tel=tel, ins=ins)
   idx = bands.index(filt)
   
   return Table([objs, RAs, DECs, res[idx], emags[idx]],
         names=['OBJ','RA','DEC','mag','emag'])

with open(os.path.join(basedir, 'PS_tcks.pkl'), 'rb') as fin:
   PS_tcks = pickle.load(fin)
def PSstand2nat(gp,rp,ip, egp=0, erp=0, eip=0, tel='SWO', ins='NC'):
   '''Take standard panstarrs g,r,i and convert to CSP ugriBV. This is
   done either through color terms (if sufficiently linear) or
   through a lookup table.'''
   if tel != 'SWO' or ins != 'NC':
      raise ValueError('telescope {} and instrument {} do not have color'\
            'terms defined'.format(tel,ins))
                       
   gmr = gp - rp; vgmr = egp**2 + erp**2
   Bcsp = splev(gmr, PS_tcks['B']) + gp
   eBcsp = sqrt(egp**2 + splev(gmr, PS_tcks['B'], 1)**2*vgmr + \
                splev(gmr, PS_tcks['eB'])**2) + Bcsp*0
   Vcsp = -0.411*(gmr) - 0.0336 + gp
   eVcsp = sqrt(0.0207**2 + egp**2 + 0.411**2*vgmr) + Vcsp*0
   ucsp = splev(gmr, PS_tcks['u']) + gp
   eucsp = sqrt(egp**2 + splev(gmr, PS_tcks['u'], 1)**2*vgmr + \
                splev(gmr, PS_tcks['eu'])**2) + ucsp*0
   gcsp = 0.0865*(gmr) + 0.027 + gp
   egcsp = sqrt(0.0212**2 + egp**2 + 0.0865**2*vgmr) + gcsp*0
   rcsp = 0.0085*(gmr) - 0.0158 + rp
   ercsp = sqrt(0.022**2 + erp**2 + 0.0085**2*vgmr) + rcsp*0
   icsp = -0.0344*(gmr) - 0.0166 + ip
   eicsp = sqrt(0.023**2 + eip**2 + 0.0344**2*vgmr) + icsp*0
   tab = Table([Bcsp,eBcsp,Vcsp,eVcsp,ucsp,eucsp,gcsp,egcsp,rcsp,ercsp,
               icsp,eicsp], names=['B','eB','V','eV','u','eu','g','eg','r','er',
                                   'i','ei'], masked=True)
   for col in tab.colnames:   
      # format and apply mask based on range of g-r from CSP LS sample
      tab[col].info.format='%.3f'
      tab[col].mask = less(gmr, -0.35) | greater(gmr, 1.56)
   return tab

def PSnat2stand(g,r,i, eg=0, er=0, ei=0, tel='SWO', ins='NC'):
   '''Take natural CSP g,r,i and convert to PANstarrs g,r,i. This is
   done through the inverse of the color terms'''
   if tel != 'SWO' or ins != 'NC':
      raise ValueError('telescope {} and instrument {} do not have color'\
            'terms defined'.format(tel,ins))
                       
   # First, convert from g-r (CSP) . to g-r (PS)
   gmr_p = (g - r - 0.027 - 0.0158)/(0.0865 - 0.0085 + 1)
   vgmr_p = eg**2 + er**2
   g_p = g - 0.027 - 0.0865*gmr_p
   eg_p = sqrt(0.0212**2 + eg**2 + 0.0865**2*vgmr_p)
   r_p = r + 0.0158 - 0.0085*gmr_p
   er_p = sqrt(0.022**2 + er**2 + 0.0158**2*vgmr_p)
   i_p = i + 0.0166 + 0.0344*gmr_p
   ei_p = sqrt(0.023**2 + ei**2 + 0.0344**2*vgmr_p)
   tab = Table([g_p,eg_p,r_p,er_p,i_p,ei_p], 
               names=['gp','egp','rp','erp','ip','eip'], masked=True)
   for col in tab.colnames:   
      # format and apply mask based on range of g-r from CSP LS sample
      tab[col].info.format='%.3f'
   return tab

with open(os.path.join(basedir, 'SM_tcks.pkl'),'rb') as fin:
   SM_tcks = pickle.load(fin)
def SMstand2nat(gp,rp,ip, egp=0, erp=0, eip=0, tel='SWO', ins='NC'):
   '''Take standard skymapper g,r,i and convert to CSP ugriBV. This is
   done either through color terms (if sufficiently linear) or
   through a lookup table.'''
   if tel != 'SWO' or ins != 'NC':
      raise ValueError('telescope {} and instrument {} do not have color'\
            'terms defined'.format(tel,ins))
                       
   gmr = gp - rp;  vgmr = egp**2 + erp**2
   Bcsp = 0.8994*gmr + 0.206 + gp
   eBcsp = sqrt(0.044**2 + egp**2 + 0.8994**2*vgmr) + Bcsp*0
   Vcsp = splev(gmr, SM_tcks['V']) + gp
   eVcsp = sqrt(0.0207**2 + egp**2 + splev(gmr, SM_tcks['V'],1)**2*vgmr) +\
         Vcsp*0
   ucsp = splev(gmr, SM_tcks['u']) + gp
   eucsp = sqrt(0.115**2 + egp**2 + splev(gmr, SM_tcks['u'],1)**2*vgmr) +\
         ucsp*0
   gcsp = 0.410*(gmr) + 0.0378 + gp
   egcsp = sqrt(0.0287**2 + 0.410**2*vgmr + egp**2) + gcsp*0
   rcsp = -0.051*(gmr) - 0.015 + rp
   ercsp = sqrt(0.024**2 + erp**2 + 0.051**2*vgmr) + rcsp*0
   icsp = -0.0473*(gmr) - 0.0166 + ip
   eicsp = sqrt(0.024**2 + eip**2 + 0.0473**2*vgmr) + icsp*0
   tab = Table([Bcsp,eBcsp,Vcsp,eVcsp,ucsp,eucsp,gcsp,egcsp,rcsp,ercsp,
               icsp,eicsp], names=['B','eB','V','eV','u','eu','g','eg','r','er',
                                   'i','ei'], masked=True)
   for col in tab.colnames:   
      # format and apply mask based on range of g-r from CSP LS sample
      tab[col].info.format='%.3f'
      tab[col].mask = less(gmr, -0.15) | greater(gmr, 1.05)

   return tab
   
def ComputeZptsFromNat(stdphot, sigclip=2, tel='SWO', ins='NC', plot=None):
   '''Given the standards.phot photometry using natural magnitudes, output by the 
   pipeline, compute the zero-point per filter, per fits file, and per night.'''

   if isinstance(stdphot, str):
      tab = ascii.read(stdphot)
   else:
      tab = stdphot

   # First, get the list of filters
   filts = list(set(tab['filt']))


   # Row:  filt, ZP, eZP, NZP, nRej, comm
   rows = []
   for filt in filts:
      stab = tab[tab['filt'] == filt] 
      # Now list of images with this filter
      fits = list(set(stab['fits']))
      zps = []
      ezps = []
      airms = []
      if plot is not None:
         fig,axes = plt.subplots(1,3, figsize=(12,5), sharey=True)
      # Plot if requested
      for k,fit in enumerate(fits):
         gids = stab['fits'] == fit
         sstab = stab[gids]
         airms.append(sstab['airm'][0])
         zp = sstab['mag'] - sstab['mins'] + kX[(tel,ins)][filt]*sstab['airm']
         ezp = sqrt(sstab['emag']**2 + sstab['emins']**2)

         avg = sum(zp*power(ezp,-2))/sum(power(ezp,-2))
         err = power(sum(power(ezp,-2)),-0.5)
         zps.append(avg)
         ezps.append(err)
         axes[0].errorbar(sstab['mag'], zp, yerr=ezp, fmt='o')
         axes[1].errorbar(sstab['airm'], zp, yerr=ezp, fmt='o', label=fit)
         axes[2].errorbar([k],[avg], yerr=[err], fmt='s')
         trans = transforms.blended_transform_factory(axes[2].transData,
               axes[2].transAxes)
         res = re.search(r'([0-9]+)', fit)
         if res is not None:
            lab = res.groups()[0]
         else:
            lab = fit
         axes[2].text(k+0.25, 0.1, lab, rotation=90, va='bottom', ha='right',
               transform=trans, color='C{}'.format(k))
         axes[0].set_xlabel('${}_{{std}}$ (mag)'.format(filt), fontsize=16)
         axes[0].set_ylabel('${}_{{ins}} - {}_{{std}} + k*airm$ (mag)'.format(
            filt,filt), fontsize=16)
         axes[1].set_xlabel('$Airmass$')
         axes[2].set_xlabel('FITS file')
         axes[2].set_xticklabels([])
      zps = array(zps)
      ezps = array(ezps)

      if len(zps) < 3:  
         rows.append([filt,None,None,len(zps),0,"N-"])
         if plot is not None:
            fig.tight_layout()
            fig.savefig('ZP_{}.pdf'.format(filt))
         continue
   
      # Some stats
      avg = mean(zps)
      med = median(zps)
      sig = std(zps)
      mad = 1.49*median(absolute(zps-med))
      print(filt,avg,med,sig,mad)

      if sig > 0.03 and len(zps) > 3:
         # Do some sigma-clipping
         for i in range(2):
            bids = greater(absolute(avg-zps),2*sig)
            avg = mean(zps[~bids])
            sig = std(zps[~bids])
         nRej = sum(bids)
      else:
         bids = None
         nRej = 0
      if plot is not None:
         # Plot the rejects
         if bids is not None:
            axes[2].plot(arange(len(zps))[bids], zps[bids], 'x', color='red',
                  zorder=100)
         for i in range(3):
            axes[i].axhline(avg, color='k')
            axes[i].axhline(avg-sig, color='k', linestyle='--', alpha=0.5)
            axes[i].axhline(avg+sig, color='k', linestyle='--', alpha=0.5)
         fig.tight_layout()
         fig.savefig('ZP_{}.pdf'.format(filt))
      # Reject non-photometric (this is Carlos' criterion)
      if (filt != 'u' and sig > 0.04) or (filt == 'u' and sig > 0.06):
         rows.append(filt,None,None,len(zps),nRej,"NP")
         continue

      rows.append([filt,avg,sig,len(zps)-nRej,nRej,'Ok'])
      table = Table(rows=rows, names=['filter','ZP','eZP','NZP','NRej','comm'])
      table['ZP'].info.format = "%.4f"
      table['eZP'].info.format = "%.4f"

   return table
      

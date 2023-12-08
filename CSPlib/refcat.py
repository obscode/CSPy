'''A pure-python module for accessing the ATLAS refcat2 catalog of
Tonry et al. (2018)i

This is basically a port of the C-code to python code. This was done because:
   1) it's more portable
   2) the tables are actually pretty small for individual queries, so
      the overhead should be minimal
'''

from astropy.io import ascii
import sys,os
from astropy.table import vstack
import numpy as np

varnames = [
  ("RA",      1e-8, "%11.7f", 1.0), 
  ("Dec",     1e-8, "%11.7f", 1.0), 
  ("plx",     1e-5/3600, "%6.2f",  3.6e6),   # mas
  ("dplx",    1e-5/3600, "%4.2f",  3.6e6),   # mas
  ("pmra",    1e-5/3600, "%8.2f",  3.6e6),   # mas
  ("dpmra",   1e-5/3600, "%4.2f",  3.6e6),   # mas
  ("pmdec",   1e-5/3600, "%8.2f",  3.6e6),   # mas
  ("dpmdec",  1e-5/3600, "%4.2f",  3.6e6),   # mas
  ("Gaia",    1e-3, "%6.3f",  1.0),
  ("dGaia",   1e-3, "%5.3f",  1.0),
  ("BP",      1e-3, "%6.3f",  1.0),
  ("dBP",     1e-3, "%5.3f",  1.0),
  ("RP",      1e-3, "%6.3f",  1.0),
  ("dRP",     1e-3, "%5.3f",  1.0),
  ("Teff",    1.0, "%5.0f",  1.0),
  ("AGaia",   1e-3, "%5.3f",  1.0),
  ("dupvar",  1,   "%1d",    1.0),
  ("Ag",      1e-3, "%5.3f",  1.0),
  ("rp1",     0.1/3600, "%4.1f",  3.6e3),  # arcsec
  ("r1",      0.1/3600, "%4.1f",  3.6e3),  # arcsec
  ("r10",     0.1/3600, "%4.1f",  3.6e3),  # arcsec
  ("g",       1e-3, "%6.3f",  1.0),
  ("dg",      1e-3, "%5.3f",  1.0),
  ("gchi",    0.01, "%5.2f",  1.0),
  ("gcontrib",1,    "%02x",   1.0),
  ("r",       1e-3, "%6.3f",  1.0),
  ("dr",      1e-3, "%5.3f",  1.0),
  ("rchi",    0.01, "%5.2f",  1.0),
  ("rcontrib",1,    "%02x",   1.0),
  ("i",       1e-3, "%6.3f",  1.0),
  ("di",      1e-3, "%5.3f",  1.0),
  ("ichi",    0.02, "%5.2f",  1.0),
  ("icontrib",1,    "%02x",   1.0),
  ("z",       1e-3, "%6.3f",  1.0),
  ("dz",      1e-3, "%5.3f",  1.0),
  ("zchi",    0.01, "%5.2f",  1.0),
  ("zcontrib",1,    "%02x",   1.0),
  ("nstat",   1,    "%3d",    1.0),
  ("J",       1e-3, "%6.3f",  1.0),
  ("dJ",      1e-3, "%5.3f",  1.0),
  ("H",       1e-3, "%6.3f",  1.0),
  ("dH",      1e-3, "%5.3f",  1.0),
  ("K",       1e-3, "%6.3f",  1.0),
  ("dK",      1e-3, "%5.3f",  1.0)
]



# Input formats 
IN_NONE = 0        # test for input format
IN_CSV  = 1        # all 44 fields from CSV refcat2
IN_BIN  = 2        # 44 field binary format created by refcat.c
IN_GRI  = 3        # just ra,dec,pmra,pmdec,rp1,r1,g,r,i,z,J

# Output formats 
OUT_ALL   = 1        # all 44 fields from refcat2 
OUT_ATLAS = 2      # just ra,dec,g,r,i,z,J,c,o 
OUT_VAR   = 3      # custom list of variables 

def RefcatQuery(ra0, dec0, rect, dra, ddec, mlim, rlim, rootdir, exten='rc2', 
      verbose=0):
   dr = np.pi/180
   # Convert coordinate degrees to radians 
   ra0 *= dr
   dec0 *= dr
   dra *= dr
   ddec *= dr

   # Identify all sqdeg overlapped by this rectangle or circle 
   # "Within rectangle" means closer in angle to the N-S, E-W great
   # circles defined by RA,Dec than the specified offsets.
   
   # The pointing 
   pointing = [np.cos(dec0) * np.cos(ra0), 
               np.cos(dec0) * np.sin(ra0), 
               np.sin(dec0)]

   # Poles of the central great circles defining the rectangle 
   rapole = [np.cos(ra0+np.pi/2), np.sin(ra0+np.pi/2), 0]
   decpole = [-np.sin(dec0) * np.cos(ra0), 
              -np.sin(dec0) * np.sin(ra0), 
              np.cos(dec0)]

   if(verbose > 1):
      sys.stderr.write("Poles for %.1f,%.1f  %6.3f,%6.3f,%6.3f are %6.3f,"\
             "%6.3f,%6.3f   %6.3f,%6.3f,%6.3f\n" % ( ra0/dr, dec0/dr, 
             pointing[0],pointing[1], pointing[2], rapole[0], rapole[1], 
             rapole[2], decpole[0], decpole[1], decpole[2]))

   if(rect):
      # Get the corners of the rectangle on the sky 
      corner = [adoffset(rapole, decpole, +dra, +ddec),
                adoffset(rapole, decpole, -dra, +ddec),
                adoffset(rapole, decpole, -dra, -ddec),
                adoffset(rapole, decpole, +dra, -ddec)]

      if(verbose > 1):
         for i in range(4):
            sys.stderr.write("Corner %d  %6.3f,%6.3f,%6.3f  %7.1f %7.1f\n" %\
                   (i, corner[i][0], corner[i][1], corner[i][2],
                   np.arctan2(corner[i][1], corner[i][0])/dr, 
                   np.arcsin(corner[i][2])/dr))

      # Dec range to consider 
      decmin = min(corner[0][2], corner[1][2])
      decmin = min(decmin, corner[2][2])
      decmin = min(decmin, corner[3][2])
      decmax = max(corner[0][2], corner[1][2])
      decmax = max(decmax, corner[2][2])
      decmax = max(decmax, corner[3][2])
      decmin = np.arcsin(decmin)
      decmax = np.arcsin(decmax)
   else:
      ddec = dra
      decmin = dec0 - ddec
      decmax = dec0 + ddec
   
   if(dec0+ddec >= np.pi/2): decmax = np.pi/2
   if(dec0-ddec <= -np.pi/2): decmin = -np.pi/2

   if(verbose > 1):
      sys.stderr.write("Dec range %.1f %.1f\n" % (decmin/dr, decmax/dr))
   

   # degin[] = 0/1 if it is inside the rectangle 
   # The original algorithem seems a bit brute-force...

   # Each corner of the rectangle lies in a sqdeg 
   inds = []
   if(rect):
      for k in range(4):
         ra = np.arctan2(corner[k][1], corner[k][0])/dr
         dec = np.arcsin(corner[k][2])/dr
         i = int(np.floor(np.fmod(ra+360,360)+1e-8))
         j = int(np.floor(dec+1e-8) + 90)
         idx = i + j*360
         if idx not in inds:  inds.append(idx)

      if(verbose > 1):
         sys.stderr.write("rectangle corners lie in:  ")
         for idx in inds:
            i = idx % 360
            j = idx//360 - 90
            sys.stderr.write(" %03d%+03d" % (i,j))
         sys.stderr.write('\n')

   # Center the circle lies in a sqdeg 
   else:
      i = int(np.floor(np.fmod(ra0/dr+360,360)+1e-8))
      j = int(np.floor(dec0/dr+1e-8) + 90)
      idx = i + j*360
      if idx not in inds: inds.append(idx)

   # dot product with rapole and decpole should be 0+/-sin{da,dd} 
   sina = np.sin(dra)
   sind = np.sin(ddec)
   # For circle dot product with pointing should be >cosda 
   cosa = np.cos(dra)

   # Mark each sqdeg that has a corner inside the rectangle or circle 
   jj = np.arange(int(np.floor(decmin/dr+1e-8)+90), 
         int(np.floor(decmax/dr-1e-8)+90)+1)
   ii = np.arange(0, 360)
   for j in jj:
      dec = (j-90) * dr
      for i in ii:
         ra = i * dr
         for k in range(4):
            P = [np.cos(dec+(k//2)*dr) * np.cos(ra+(k%2)*dr),
                 np.cos(dec+(k//2)*dr) * np.sin(ra+(k%2)*dr),
                 np.sin(dec+(k//2)*dr)]
            if(rect):
               if(np.dot(P,pointing) < 0 or
                  np.dot(P,rapole) > sina  or np.dot(P,rapole) < -sina or
                  np.dot(P,decpole) > sind or np.dot(P,decpole) < -sind):
                  continue
            else:
               if(np.dot(P,pointing) < cosa): continue
            print(i,j)
            idx = i + j*360
            if idx not in inds: inds.append(idx)
            break
   
   if(verbose > 1):
      sys.stderr.write("Sqdeg with corners inside area:  ")
      for idx in inds:
         i = idx % 360
         j = idx//360 - 90
         print("    ",i,j)
         sys.stderr.write(" %03d%+03d" % (i, j))
      sys.stderr.write("\n")

   # Read each refcat file, keep desired stars 
   stack = []
   for d in rootdir:
      for idx in inds:
         i = idx % 360
         j = idx//360 - 90
         fname = "%s/%03d%+03d.%s" % (d, i, j, exten)
         if rect:
            tab = read_csv(fname, mlim, rlim, rect, rapole, sina, decpole, sind)
         else:
            tab = read_csv(fname, mlim, rlim, rect, pointing, cosa, decpole,
                  sind)
         if(verbose > 0):
            sys.stderr.write("Read file %s with format %s total number of "\
                  "stars %d\n" % (fname, "csv", len(tab)))
         stack.append(tab)

   # Combine tables
   tab = vstack(stack)

   return tab

# Sky coord defined by offset of da,dd[rad] from the circles through a,d 
# p.a=sin(da) and a.z=0   p.d=sin(dd)   p.p=1 
def adoffset(a, d, da, dd):
   '''Sky coord defined by offset of da,dd (in rad) form the circles through
   a,d.'''

   if np.absolute(da) + np.absolute(dd) >= np.pi/2:
      return(0.,0.,0.)

   # Solve for point when RA=Dec=0, i.e. a=y and d=z 
   py = np.sin(da)
   pz = np.sin(dd)
   px = np.sqrt(1 - py**2 - pz**2)
   
   # Rotate by -Dec around y and by RA around z 
   # Note sin(RA)=-a.x, cos(RA)=a.y; sin(Dec)=-d.x/a.y, cos(Dec)=d.z 
   sa = -a[0]
   ca =  a[1]
   sd = np.where(np.absolute(ca) > 0.7, -d[0]/ca, -d[1]/sa)
   cd = d[2]

   tmp = px
   px = px * cd - pz * sd
   pz =  tmp * sd + pz * cd

   tmp = px
   px = px * ca - py * sa
   py =  tmp * sa + py * ca
   return (px,py,pz)

# Read all 44 columns from CSV file 
def read_csv(fname, mlim, rlim, rect, p1, t1, p2, t2):

   dr=np.pi/180

   converters = {}
   # Deal with hex-format for bitmaps
   for f in ['g','r','i','z']:
      converters[f+'contrib'] = [ascii.convert_numpy(str)]

   tab = ascii.read(fname, names=[v[0] for v in varnames],
         converters=converters)
   for name,fac,fmt,scale in varnames:
      if name.find('contrib') > 0: continue
      tab[name] = tab[name]*fac
      tab[name].info.format = fmt

   # Is it bright enough? 
   m = np.minimum.reduce([tab['g'],tab['r'],tab['i']])
   gids = np.less_equal(m, mlim)

   # Is it isolated enough? 
   gids = gids*np.greater_equal(tab['rp1'], rlim/3600)

   # Is it inside the rectangle or radius?  Skip if not. 
   P = [np.cos(tab['Dec']*dr) * np.cos(tab['RA']*dr),
        np.cos(tab['Dec']*dr) * np.sin(tab['RA']*dr),
        np.sin(tab['Dec']*dr)]
   Pp1 = np.dot(p1,P)
   if rect:
      Pp2 = np.dot(p2,P)
      gids = (Pp1 <= t1)*(Pp1 >= -t1)*(Pp2 <= t2)*(Pp2 >= -t2) 
   else:
      gids = np.greater(Pp1, t1)

   return tab[gids]


if __name__ == "__main__":
   import argparse
   import os

   parser = argparse.ArgumentParser(description=\
     "Returns stars close to ra, dec from ATLAS Refcat2 sqdeg files.")
   parser.add_argument('ra', help="RA in decimal degrees", type=float)
   parser.add_argument('dec', help="DEC in decimal degrees", type=float)
   parser.add_argument('-dir', help="Read the data files from directory D",
         default="/atlas/cal/RC2/m17")
   parser.add_argument('-exten', help="Data file have file names P/rrr+dd.X",
         default="rc2")
   parser.add_argument('-infmt', help="Requst refcat to read from CSV text or"\
         " binary files of the form P/rrr+dd.X. The default is to attempt "\
         "auto-detect the file type", type=str, choices=['none','csv','bin'],
         default='none')
   parser.add_argument('-mlim', help="Return only stars with the smallest of"\
         " g,r,i less than or equal to m.", type=float, default=18.0)
   parser.add_argument('-all', help="Request refcat to return all 44 fields "\
         "from Refcat2 according to the units given in the Value column of the"\
         " table in the man page.", action="store_true")
   parser.add_argument('-var', help="Output custom fields. Separate each with"\
         " a comma. Example:  -var RA,Dec,g,r,i")
   parser.add_argument('-rect', help="Return stars within a rectagle centered"\
         " at RA,DEC with width dR and height dD", nargs=2, default=[0.1,0.1],
         type=float)
   parser.add_argument('-rad', help="Return stars within a rectagle centered"\
         " at RA,DEC with radius RAD", type=float, default=0)
   parser.add_argument('-verb', help="Be verbose", action='store_true')
   parser.add_argument('-VERB', help='Maximum verbosity', action='store_true')

   args = parser.parse_args()

   rootspec = args.dir      # Root directory
   mlim = args.mlim         # Limiting magnitude for m<mlim
   rlim = 0.0               # Limiting radius for rp1>rlim (hidden option???)

   VERBOSE=0 
   if args.verb:  VERBOSE=1
   if args.VERB:  VERBOSE=2

   infmt = args.infmt       # Test for format
   outfmt = OUT_ATLAS       # Output format
   if args.all: 
      outfmt=OUT_ALL
   elif args.var is not None:
      outfmt=OUT_VAR

   exten = args.exten        # Input file extension
   # If user specifies rad, use that
   if args.rad > 0:
      dra = args.rad
      ddec = 0
      rect = 0
   else:
      dra,ddec = args.rect
      rect = 1

   # Pick apart the rootspec into directories 
   rootdir = rootspec.split(',')
   ndir = len(rootdir)
   for d in rootdir:
      if not os.path.isdir(d):
         raise IOError('Cannot access root directory {}'.format(d))

   if(VERBOSE > 0):
      print("Searching directories:")
      for i,d in enumerate(rootdir):
         print("{} {}".format(i,d))

   # More sanity checks 
   if dra == 0 or (rect and ddec == 0):
      raise ValueError("Require a radius or rectangle dimension")

   tab = RefcatQuery(args.ra, args.dec, rect, dra, ddec, mlim, rlim, rootdir, 
         exten)

   if (outfmt == OUT_ATLAS):

      # Synthesize (181023) ATLAS cyan and orange from a couple of observations
      # 02a58400o0400c and 02a58406o0400o.  There's a bit of curvature:
      #(g - c_inst) = 25.77 + 0.467 (g-r) + 0.048 (g-r)^2^
      #(r - o_inst) = 25.59 + 0.443 (r-i) + 0.090 (r-i)^2^

      # Write ATLAS-specific results to stdout 
      clr = tab['g'] - tab['r']
      tab['c'] = tab['g'] - 0.467*clr - 0.048*clr**2
      tab['c'].info.format = "%6.3f"
      clr = tab['r'] - tab['i']
      tab['o'] = tab['r'] - 0.443*clr - 0.090*clr**2
      tab['o'].info.format = "%6.3f"
      tab = tab['RA','Dec','g','r','i','z','J','c','o']
      tab.write(sys.stdout, format='ascii.fixed_width', delimiter=' ')

   # Dump the entire star record 
   elif (outfmt == OUT_ALL):
      tab.write(sys.stdout, format='ascii.fixed_width', delimiter=' ')

   # Dump a custom list of variables 
   elif (outfmt == OUT_VAR):
      varlist = args.var.split(',')
      for var in varlist:
         if var not in tab.colnames:
            raise ValueError('Error:  field {} not recognized. Should'\
                ' be one of {}'.format(var, ','.join([v[0] for v in varnames])))
      tab = tab[varlist]
      tab.write(sys.stdout, format='ascii.fixed_width', delimiter=' ')

'''Format of Refcat2 CSV file 
Col Varname  Entry    Units         Value        Description
 1  RA   28000001672 [10ndeg]  280.00001672~deg  RA from Gaia DR2, J2000, epoch 2015.5
 2  Dec  -1967818581 [10ndeg]  -19.67818581~deg  Dec from Gaia DR2, J2000, epoch 2015.5
 3  plx        98    [10uas]        0.98~mas     Parallax from Gaia DR2
 4  dplx       10    [10uas]        0.10~mas     Parallax uncertainty
 5  pmra      114    [10uas/yr]     1.14~mas/yr  Proper motion in RA from Gaia DR2
 6  dpmra      16    [10uas/yr]     0.16~mas/yr  Proper motion uncertainty in RA
 7  pmdec   -1460    [10uas/yr]   -14.60~mas/yr  Proper motion in Dec from Gaia DR2
 8  dpmdec     15    [10uas/yr]     0.15~mas/yr  Proper motion uncertainty in Dec
 9  Gaia    15884    [mmag]        15.884        Gaia DR2 G magnitude
10  dGaia       1    [mmag]         0.001        Gaia DR2 G magnitude uncertainty
11  BP      16472    [mmag]        16.472        Gaia G_BP magnitude
12  dBP        10    [mmag]         0.010        Gaia G_BP magnitude uncertainty
13  RP      15137    [mmag]        15.137        Gaia G_RP magnitude
14  dRP         1    [mmag]         0.001        Gaia G_RP magnitude uncertainty
15  Teff     4729    [K]           4729~K        Gaia stellar effective temperature
16  AGaia     895    [mmag]         0.895        Gaia estimate of G-band extinction for this star
17  dupvar      2    [...]          2            Gaia flags coded as CONSTANT (0), VARIABLE (1), or NOT_AVAILABLE (2) + 4*DUPLICATE
18  Ag       1234    [mmag]         1.234        SFD estimate of total column g-band extinction
19  rp1        50    [0.1asec]      5.0~arcsec   Radius where cumulative G flux exceeds 0.1x this star
20  r1         50    [0.1asec]      5.0~arcsec   Radius where cumulative G flux exceeds 1x this star
21  r10       155    [0.1asec]     15.5~arcsec   Radius where cumulative G flux exceeds 10x this star
22  g       16657    [mmag]        16.657        Pan-STARRS g_P1 magnitude
23  dg         10    [mmag]         0.010        Pan-STARRS g_P1 magnitude uncertainty
24  gchi       23    [0.01]         0.23         chi^2/DOF for contributors to g
25  gcontrib   1f    [%02x]       00011111       Bitmap of contributing catalogs to g
26  r       15915    [mmag]        15.915        Pan-STARRS r_P1 magnitude
27  dr         12    [mmag]         0.012        Pan-STARRS r_P1 magnitude uncertainty
28  rchi       41    [0.01]         0.41         chi^2/DOF for contributors to r
29  rcontrib   3f    [%02x]       00111111       Bitmap of contributing catalogs to r
30  i       15578    [mmag]        15.578        Pan-STARRS i_P1 magnitude
31  di         10    [mmag]         0.010        Pan-STARRS i_P1 magnitude uncertainty
32  ichi       49    [0.01]         0.49         chi^2/DOF for contributors to i
33  icontrib   0f    [%02x]       00001111       Bitmap of contributing catalogs to i
34  z       15346    [mmag]        15.346        Pan-STARRS z_P1 magnitude
35  dz         12    [mmag]         0.012        Pan-STARRS z_P1 magnitude uncertainty
36  zchi        0    [0.01]         0.00         chi^2/DOF for contributors to z
37  zcontrib   06    [%02x]       00000110       Bitmap of contributing catalogs to z
38  nstat       0    [...]          0            Count of griz deweighted outliers
39  J       14105    [mmag]        14.105        2MASS J magnitude
40  dJ         36    [mmag]         0.036        2MASS J magnitude uncertainty
41  H       14105    [mmag]        14.105        2MASS H magnitude
42  dH         53    [mmag]         0.053        2MASS H magnitude uncertainty
43  K       13667    [mmag]        13.667        2MASS K magnitude
44  dK         44    [mmag]         0.044        2MASS K magnitude uncertainty
#//////////////////////////////////////////////////////////////
'''

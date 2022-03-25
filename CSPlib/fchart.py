import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from astropy.visualization import simple_norm
from astropy.wcs import WCS

'''Fchart.py:   Make a finder chart with cut-out closeup and offset stars or
catalog stars.'''

symbs = ['s', 'o', 'd', '^', 'v','<','>'][::-1]

def Fchart(fts, percent=99, maxpercent=None, minpercent=None,
      offsetcat=None, LScat=None, zoomfac=4, snx='SNX', sny='SNY',
      sn=None):
   '''Draw a finder chart for the given FITS image. 

   Args:
      fts (astropy.fits):  the FITS instance for which to draw the chart.
                           It should have a valid WCS.
      percent (float): The percent of pixel values to display in auto-scale
      max_percent (float): The percent of pixels to display at the hight end
      min_percent (float): The percent of pixels to display at the low end
      offsetcat (astropy.table):  Catalog of offset stars
      LScat (astropy.table):  Catalog of local sequence stars
      zoomfac (int):   The zoom factor for the cutout
      snx,sny (float/str): Header keyword or value of the SN position (degrees)
      sn (str): SN name. If not specified, get it from the header
   
   Returns:
      matplotlib.figure instance:  the finder chart
   '''

   if sn is None:
      if 'OBJECT' not in fts[0].header:
         sn = 'Unknown'
      else:
         sn = fts[0].header['OBJECT']

   fig = plt.figure(figsize=(9,9))

   wcs = WCS(fts[0])
   ax = fig.add_subplot(111, projection=wcs)
   plt.subplots_adjust(left=0.2)
   norm = simple_norm(fts[0].data, percent=percent, 
         max_percent=maxpercent, min_percent=minpercent)
   ax.imshow(fts[0].data, origin='lower', norm=norm, cmap='gray_r')

   jsize,isize= fts[0].data.shape
   # Figure out SN position and scale of image
   if isinstance(snx, str):
      if snx not in fts[0].header:
         # Assume center of frame
         isn = isize/2
         jsn = jsize/2
         xsn,ysn = wcs.wcs_pix2world(isn,jsn,0)
      else:
         xsn= fts[0].header[snx]
         ysn = fts[0].header[sny]
         isn,jsn = wcs.wcs_world2pix(xsn,ysn,0)
   else:
      xsn = snx
      ysn = sny
      isn,jsn = wcs.wcs_world2pix(xsn,ysn,0)

   x0,y0 = wcs.wcs_pix2world(0,0,0)
   x1,y1 = wcs.wcs_pix2world(isize,jsize,0)
   xsize = abs(x1-x0)
   ysize = abs(y1-y0)

   # Cental cross-hair
   ax.plot([isn - 0.2*isize, isn - 0.02*isize],[jsn, jsn], '-', color='red', 
         alpha=0.5)
   ax.plot([isn + 0.02*isize, isn + 0.2*isize],[jsn, jsn], '-', color='red', 
         alpha=0.5)
   ax.plot([isn,isn],[jsn - 0.2*jsize, jsn - 0.02*jsize], '-', color='red', 
         alpha=0.5)
   ax.plot([isn,isn],[jsn + 0.02*jsize, jsn + 0.2*jsize], '-', color='red', 
         alpha=0.5)
   ax.set_xlabel('RA (J2000)')
   ax.set_ylabel('DEC (J2000)')

   # Plot offset-stars, if specifid
   if offsetcat is not None:
      cat = offsetcat
      for i in range(len(cat)):
         dx = np.cos(ysn*np.pi/180)*(cat['RA'][i]-xsn)*3600
         dy = (cat['DEC'][i]-ysn)*3600
         ii,jj = wcs.wcs_world2pix(cat['RA'][i], cat['DEC'][i], 0)
         ax.plot(ii, jj, symbs.pop(), mec='blue', mfc='none', ms=20,
               label="({:.1f},{:.1f})".format(dx,dy))
      ax.legend(loc='upper left', fontsize=10, markerscale=0.5)

   # Plot LS-stars if specified
   if LScat is not None:
      cat = LScat
      if 'objID' not in cat.colnames:
         cat['objID'] = arange(len(cat))

      ii,jj = wcs.wcs_world2pix(cat['RA'], cat['DEC'], 0)
      ax.plot(ii, jj, 'o', mec='blue', mfc='none', ms=20)
      for i in range(len(cat)):
         ax.text(ii[i]+isize*0.02,jj[i]+jsize*0.02, cat['objID'][i], 
               va='bottom', ha='left', fontsize=10)

   # Cut-out
   if zoomfac is not None and zoomfac > 0:
      ins = inset_axes(ax, width="100%", height="100%",
            bbox_to_anchor=(0.7,0.7, 0.4, 0.4), bbox_transform=ax.transAxes)
      ins.tick_params(left=False, right=False, bottom=False, top=False)
      ins.axes.get_xaxis().set_visible(False)
      ins.axes.get_yaxis().set_visible(False)
      size = max(isize,jsize)/zoomfac
      xx0 = int(isn - size/2)
      xx1 = int(isn + size/2)
      yy0 = int(jsn - size/2)
      yy1 = int(jsn + size/2)
 
      subdata = fts[0].data[yy0:yy1,xx0:xx1]
      ins.imshow(subdata,origin='lower', norm=norm, cmap='gray_r')
 
      #centered on SN, so no need for trans.
      ins.plot([0.25,0.45],[0.5, 0.5], '-', color='red', alpha=0.5,
         transform=ins.transAxes)
      ins.plot([0.55,0.75],[0.5, 0.5], '-', color='red', alpha=0.5,
         transform=ins.transAxes)
      ins.plot([0.5,0.5],[0.25, 0.45], '-', color='red', alpha=0.5,
         transform=ins.transAxes)
      ins.plot([0.5,0.5],[0.55, 0.75], '-', color='red', alpha=0.5,
         transform=ins.transAxes)

   # Compass
   ax.plot([0.95,0.85],[0.05,0.05], '-', color='blue', transform=ax.transAxes)
   ax.plot([0.95,0.95],[0.05,0.15], '-', color='blue', transform=ax.transAxes)
   ax.text(0.84, 0.05, 'E', fontsize=16, color='blue', transform=ax.transAxes,
         ha='right', va='center')
   ax.text(0.95, 0.17, 'N', fontsize=16, color='blue', transform=ax.transAxes,
         ha='center', va='bottom')


   ax.set_title(sn, loc='left')

   # Scale
   i0,j0 = xsize*0.05,ysize*0.05
   x0,y0 = wcs.wcs_pix2world(i0, j0, 0)
   i1, j1 = wcs.wcs_world2pix(x0 - 1/60, y0, 0)
   ax.plot([i0,i1],[j0,j1], '-', color='blue')
   ax.text((i0+i1)/2, j0-10, "1'", color='blue', ha='center', va='top')

   return fig

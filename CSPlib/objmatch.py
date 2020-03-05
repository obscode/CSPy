'''Given two sets of coordinates, match up sets of objects'''
from numpy import *
from .npextras import bwt 
from .basis import svdfit,abasis
from .sextractor import SexTractor
from astropy.wcs import WCS
from astropy.io import fits

def fitscalerot(x0, y0, x1, y1):
   '''compute a scale+rotation transformation.'''
   basis = abasis(0, x0, y0, rot=1)
   sol = svdfit(basis, concatenate([x1,y1]))
   ixy = add.reduce(sol[newaxis,:]*basis,1)
   ix,iy = ixy[:len(ravel(x0))], ixy[len(ravel(x0)):]
   theta = arctan2(sol[3],sol[2])
   scale = sol[2]/cos(theta)
   xshift,yshift = sol[0:2]
   return xshift,yshift,scale,theta,ix,iy,sol

def fitpix2RADEC(i, j, x, y):
   '''compute a WCS via CDn_n pixel matrix.'''
   I = ones(i.shape)
   Z = zeros(i.shape)
   sb = [I, Z, -i, j]
   eb = [Z, I, j, i]
   sb = transpose(sb); eb = transpose(eb)
   basis = concatenate([sb,eb])
   sol = svdfit(basis, concatenate([x,y]))
   xshift,yshift = sol[0:2]
   cd11 = -sol[2]
   cd12 = cd21 = sol[3]
   cd22 = sol[2]
   return xshift,yshift, cd11, cd12, cd21, cd22


def objmatch(x1,y1,x2,y2, dtol, atol, scale1=1.0, scale2=1.0, 
      angles=[0], verb=False):

   dx1 = (x1[newaxis,:] - x1[:,newaxis]).astype(float32)*scale1
   dx2 = (x2[newaxis,:] - x2[:,newaxis]).astype(float32)*scale2
   dy1 = (y1[newaxis,:] - y1[:,newaxis]).astype(float32)*scale1
   dy2 = (y2[newaxis,:] - y2[:,newaxis]).astype(float32)*scale2
   da1 = arctan2(dy1,dx1)*180/pi
   da2 = arctan2(dy2,dx2)*180/pi
   ds1 = sqrt(power(dx1,2) + power(dy1,2))
   ds2 = sqrt(power(dx2,2) + power(dy2,2))
   del dx1
   del dx2
   del dy1
   del dy2
   best_N = -1
   best_a = 0
   ds = ds1[:,:,newaxis,newaxis]-ds2[newaxis,newaxis,:,:]
   for aoff in angles:
      da = da1[::,::,newaxis,newaxis] - da2[newaxis,newaxis,::,::] + aoff
      da = where(less(da, -180.0), da+180.0, da)
      da = where(greater(da, 180.0), da-180.0, da)
      use = less(absolute(ds),dtol)*less(absolute(da),atol)
      suse = add.reduce(add.reduce(use,3),1)
      if max(ravel(suse)) < 4: 
         if verb: print("angle {:.2f} gives < 4 matches".format(aoff))
         continue
      guse = greater(suse,max(suse.ravel())/2)
      if verb: print("angle {:.2f} gives {} matches".format(aoff, 
            sum(ravel(guse))))
      if sum(ravel(guse)) > best_N:
         best_a = aoff
         best_N = sum(ravel(guse))
   da = da1[::,::,newaxis,newaxis] - da2[newaxis,newaxis,::,::] + best_a
   da = where(less(da, -180.0), da+180.0, da)
   da = where(greater(da, 180.0), da-180.0, da)
   use = less(absolute(ds),dtol)*less(absolute(da),atol)
   suse = add.reduce(add.reduce(use,3),1)
   guse = greater(suse,max(suse.ravel())/2)
   if verb:
      print("Found {} matches".format(sum(ravel(guse))))
      if best_a != 0:
         print("Best angular offset = {:.2f}".format(best_a))

   i = [j for j in range(x1.shape[0]) if sum(guse[j])]
   m = [argmax(guse[j]) for j in range(x1.shape[0]) if sum(guse[j])]
   xx1,yy1 = take([x1,y1],i,1)
   xx2,yy2 = take([x2,y2],m,1)
   rscale = scale2/scale1
   best_a = best_a*pi/180.0
   xt2 = rscale*(cos(best_a)*xx2 + sin(best_a)*yy2)
   yt2 = rscale*(-sin(best_a)*xx2 + cos(best_a)*yy2)
   xshift,xscat = bwt(xx1-xt2)
   xscat = max([1.0,xscat])
   yshift,yscat = bwt(yy1-yt2)
   yscat = max([1.0,yscat])
   print(xscat,yscat)
   keep = less(absolute(xx1-xt2-xshift),3*xscat)*\
          less(absolute(yy1-yt2-yshift),3*yscat)
   xx1,yy1,xx2,yy2 = compress( keep, [xx1,yy1,xx2,yy2], 1)
   #wt = ones(x0.shape,float32)
   return xshift,yshift,xx1,yy1,xx2,yy2

def iterativeSol(x1, y1, x2, y2, scale1=1.0, scale2=2.0, dtol=1.0, atol=1.0, 
      angles=[0], Niter=3, verb=False):
   '''Using iteration, solve for the transformation from x1,y1 to
   x2,y2, returning the solution and tranformed x1,y1

   Args:
      x1,y1 (float,float):  coordinates to be transformed
      x2,y2 (float,float):  target coordinates
      scale1 (float):  pixel scale of (x1,y1)
      scale2 (float):  pixel scale of (x2,y2)
      dtol (float):  matching distance tolorance in scaled coordinates
      atol (float):  matching angles in degrees
      angles (list of floats):  Try these angle offsets between the two
                                coordinate systems.
      Niter (int):  Number of iterations
      verb (bool):  Be verbose?

   Returns:
      sol,xt,yt:   sol = array of floats, the solution
                   xt,yt = transformed x1, y1
   '''

   # First get matching set
   xshift,yshift,xx1,yy1,xx2,yy2 = objmatch(x1,y1,x2,y2,dtol, atol,
         scale1, scale2, angles, verb)

   if len(xx1) < 3:
      if verb:
         print("Sorry, less than 3 matches found, giving up")
      return None,None,None

   for iter in range(Niter):
      if iter:   # after first iteration
         basis = abasis(0, x1, y1, rot=1)
         ixy = add.reduce(sol[newaxis,:]*basis, 1)
         ix,iy = ixy[:len(ravel(x1))], ixy[len(ravel(x1)):]
         delx = ix[:,newaxis] - x2[newaxis,:]
         dely = iy[:,newaxis] - y2[newaxis,:]
         dels = sqrt(power(delx,2) + power(dely,2))
         ui0 = [j for j in range(delx.shape[0]) if min(dels[j]) < dtol]
         ui1 = [argmin(dels[j]) for j in range(delx.shape[0]) \
                if min(dels[j]) < dtol]
         if len(ui0) == 0:
            if verb:
               print("Error:  residuals of coordinate transformation are all "
                     "greater than dtol")

            return None,None,None
         xx1,yy1 = take([x1, y1], ui0, 1)
         xx2,yy2 = take([x2, y2], ui1, 1)
      if verb:
         print("Pass {} with {} objects.".format(iter+1, len(x1)))
      xshift,yshift,scale,rot,ix,iy,sol = fitscalerot(xx1,yy1,xx2,yy2)
      print(xshift,yshift, scale, rot)
      delx = ix-xx2
      dely = iy-yy2
      dels = sqrt(power(delx,2) + power(dely,2))
      scx = bwt(delx)[1]
      scy = bwt(dely)[1]
      if verb:
         print("Biweight estimate for scatter in coordinate trans: (x,y) ="
            "({:.5f},{:.5f})".format(scx,scy))
   return sol, ui0, ui1


def WCStoImage(wcsimage, image, scale='SCALE', tel='SWO', 
      ins='NC', Nstars=50, verbose=False, angles=[0.0]):
   '''Given a FITS image with a WCS, solve for the WCS in a different
   image.

   Args:
      wcsimage (str or FITS):  Image with the WCS
      image (str or FITS):  Image to solve
      scale (str of float): The plate scale of the images to solve.
                            if a string, get scale from the specified 
                            header keyword.
      tel (str):   Telescope code (e.g., SWO)
      ins (str):   Instrument code (e.g., NC)
   Returns:
      The original image with WCS header information updated.

   Note:
      We currently only solve for a shift, scale and rotation
   '''

   if isinstance(wcsimage, str):
      wcsimage = fits.open(wcsimage)
   if isinstance(image, str):
      image = fits.open(image)
   if isinstance(scale, str):
      imscale = image[0].header[scale]

   # get scale from WCS, since we have it
   wcs = WCS(wcsimage[0])
   wscale = abs(wcs.pixel_scale_matrix[0,0])*3600   # in arc-sex/pixel

   s = SexTractor(image, tel, ins)
   s.run()
   icat = s.parseCatFile()
   s.cleanup()
   gids = icat['FLAGS'] < 1
   icat = icat[gids]
   icat = icat[argsort(icat['MAG_APER'])]

   s = SexTractor(wcsimage, gain=1.0, scale=wscale)
   s.run()
   wcat = s.parseCatFile()
   s.cleanup()
   gids = wcat['FLAGS'] < 1
   wcat = wcat[gids]
   wcat = wcat[argsort(wcat['MAG_APER'])]

   icat = icat[:Nstars] 
   wcat = wcat[:Nstars]

   #x0,y0 = wcs.wcs_pix2world(wcat['X_IMAGE'], wcat['Y_IMAGE'], 0)
   x1,y1 = wcat['X_IMAGE'], wcat['Y_IMAGE']
   x0,y0 = icat['X_IMAGE'],icat['Y_IMAGE']

   res,idx1,idx2 = iterativeSol(x0, y0, x1, y1, scale1=imscale, scale2=wscale,
         dtol=1.0, verb=verbose, angles=angles)
   if res is None: return None
   ii,ij = take([x0,y0], idx1, 1)
   wi,wj = take([x1,y1], idx2, 1)

   x,y = wcs.wcs_pix2world(wi,wj,0)
   # Now solve or CD matrix
   crval1,crval2,cd11,cd12,cd21,cd22 = fitpix2RADEC(ii, jj, x, y)

   image[0].header['CRPIX1'] = 1   # FITS standard indexes from 1
   image[0].header['CRPIX2'] = 1
   image[0].header['CRVAL1'] = crval1
   image[0].header['CRVAL2'] = crval2
   image[0].header['CD1_1'] = cd11
   image[0].header['CD1_2'] = cd12
   image[0].header['CD2_1'] = cd21
   image[0].header['CD2_2'] = cd22

   return fitpix2RADEC(ii,ij,x,y)

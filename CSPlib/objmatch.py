'''Given two sets of coordinates, match up sets of objects'''
from numpy import *
from .npextras import bwt

def objmatch(x1,y1,x2,y2, dtol, atol, verb=False):
   dx1 = (x1[newaxis,:] - x1[:,newaxis]).astype(float32)
   dx2 = (x2[newaxis,:] - x2[:,newaxis]).astype(float32)
   dy1 = (y1[newaxis,:] - y1[:,newaxis]).astype(float32)
   dy2 = (y2[newaxis,:] - y2[:,newaxis]).astype(float32)
   da1 = arctan2(dy1,dx1)*180/pi
   da2 = arctan2(dy2,dx2)*180/pi
   dx = dx1[::,::,newaxis,newaxis] - dx2[newaxis,newaxis,::,::]
   dy = dy1[::,::,newaxis,newaxis] - dy2[newaxis,newaxis,::,::]
   ds = sqrt(dx**2+dy**2)
   del dx1
   del dx2
   del dy1
   del dy2
   da = da1[::,::,newaxis,newaxis] - da2[newaxis,newaxis,::,::]
   use = less(absolute(ds),dtol)*less(absolute(da),atol)
   suse = add.reduce(add.reduce(use,3),1)
   if verb:
      for suse1 in suse: print(suse1)
   guse = greater(suse,max(suse.ravel())/2)
   i = [j for j in range(x1.shape[0]) if sum(guse[j])]
   m = [argmax(guse[j]) for j in range(x1.shape[0]) if sum(guse[j])]
   xx1,yy1 = take([x1,y1],i,1)
   xx2,yy2 = take([x2,y2],m,1)
   xshift,xscat = bwt(xx1-xx2)
   #xscat = max([1.0,xscat])
   yshift,yscat = bwt(yy1-yy2)
   #yscat = max([1.0,yscat])
   keep = less(abs(xx1-xx2-xshift),3*xscat)*less(abs(yy1-yy2-yshift),3*yscat)
   xx1,yy1,xx2,yy2 = compress( keep, [xx1,yy1,xx2,yy2], 1)
   #wt = ones(x0.shape,float32)
   return xshift,yshift,xx1,yy1,xx2,yy2


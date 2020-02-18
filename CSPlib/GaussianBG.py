from numpy import *
from scipy.optimize import leastsq

def between(x, l, h, inc=0):
   if inc: return greater_equal(x,l)*less_equal(x,h)
   else: return greater(x,l)*less(x,h)

def getpct(x,q,axis=0):
   '''Return the percentile of an array'''
   x=asarray(x)
   if len(x.shape)> 1: x=sort(x,axis)
   else: x=sort(x)
   j = min([len(x)-1,int(0.01*q*len(x))])
   if len(x) > 2:
      if axis == 0: return x[j]
      else: return x[::,j]
   elif len(x): return sum(x)/(1.0*len(x))
   else: return 0.0

def getsqrt(x,rep=0):
    y = less_equal(x,0)
    y2 = isfinite(x)
    y = logical_and(y,y2)
    z = multiply(x,logical_not(y))
    return sqrt(z)+rep*y

def extrema(x):
   return asarray([minimum.reduce(x), maximum.reduce(x)])

def divz(x,y=1,repl=0.0,out=None):
   if len(shape(y)) or len(shape(x)):
      if len(shape(y)): bad = equal(y,0.0)
      else: bad = ones(x.shape)*(y==0.0)
      not_bad = ~bad
      numer = (x*not_bad)
      denom = (y+bad)
      a = (repl*bad)
      if out: a = a.astype(out)
      b = (numer/denom)
      if out: b = b.astype(out)
      c = (a + b)
      if out: c = c.astype(out)
      return c
   else:
      if y == 0: return repl
      else: return x/y

def getexp(x,flo=-1000,fhi=1000):
    x2 = x.astype(longfloat)
    y = greater(x2,flo)*less(x2,fhi)
    y2 = isfinite(x2)
    y = logical_and(y,y2)
    g = exp(y*x2)*y
    g = where(isnan(g),0.0,g).astype(x.dtype)
    return g

def fitgaussian(p,x,y,f=0,v=0,wt=None):
   c,s = p
   s = pow(10,s)
   k = -0.5*power((x-c)/s,2)
   g = getexp(k)
   u = 1.0*greater(y,median(y))*greater(g,1e-4)
   if wt: w = wt
   else: w = divz(u,getsqrt(y))
   a = sum(compress(w,divz(w*y,g)))/sum(compress(w,w))
   m = a*g
   u = 1.0*logical_or(greater(x,c)*greater(y,median(y)),less_equal(x,c))
   u = u * greater(m,0)
   if wt: w = wt
   else: w = divz(u,getsqrt(y))
   if f:
     r = (y-m)*u*w
     if v > 1: print(a,c,s,add.reduce(r**2))
     return r
   else: return m

def getgausky(D,dn,verb=0):
   D = sort(D)
   p40 = D[40*len(D)//100+1]
   p10 = D[10*len(D)//100+1]
   se = 1.49*median(abs(D-p40))
   kD = between(D,p40-4*se,p40+3*se)
   uD = compress(kD,D)
   b = diff(extrema(uD))[0]/dn
   H = arange(min(uD),max(uD)+b/2.,b,float64)
   dS = diff(searchsorted(uD,H))
   H = diff(H)/2. + H[:-1]
   guess = [p40,log10(se)]

   if verb > 1: print("Fitting:",guess)
   H,dS = compress(greater(dS,0),[H,dS],1)
   #if verb >= 3:
   #   from pgplot import newbox,xlabel,ylabel,drawln
   #   newbox(extrema(H),[-0.1*max(dS),1.2*max(dS)],ch=2)
   #   xlabel("H",ch=2)
   #   ylabel("dS",ch=2)
   #   drawln(H,dS,hist=1,ls=1)
   gsol,gmsg = leastsq(fitgaussian,guess,args=(H,dS,1,verb),xtol=1e-6,ftol=1e-6,
                       epsfcn=1e-7,factor=10.0,maxfev=5000)
   gsol[1] = 10**gsol[1]
   if verb >= 2: print(" %9.1f %9.1f" % tuple(gsol))
   #if verb >= 3:
   #    drawln([gsol[0],gsol[0]],[-max(dS),10*max(dS)],ls=2,ci=7)
   #    drawln([gsol[0]+gsol[1],gsol[0]+gsol[1]],[-max(dS),10*max(dS)],ls=4,ci=7)
   #    drawln([gsol[0]-gsol[1],gsol[0]-gsol[1]],[-max(dS),10*max(dS)],ls=4,ci=7)
   return gsol[0]

def GaussianBG(data,dn,verb=0,check=1):
   if check: 
      data = compress(greater(abs(data.ravel()),abs(getpct(data.ravel(),66))/1e12),
                      data.ravel())
   data = sort(data)
   bg = getgausky(data.ravel(),dn,verb=verb)
   return bg

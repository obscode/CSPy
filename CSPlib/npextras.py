'''A collection of numpy-based tools and statistical measures.'''
from numpy import *

def between(x, l, h, inc=0):
   '''Return boolean array of values between bounds.

   Args:
      x (array):  array to work on
      l (float):  lower bound
      h (float):  upper bound
      inc (bool):  Inclusive bounds?
   
   Returns:
      boolean array with same shape as x
   '''

   if inc: return greater_equal(x,l)*less_equal(x,h)
   else: return greater(x,l)*less(x,h)


def getsqrt(x,rep=0):
   '''Compute square root, replacing invalid results.

   Args:
      x (array):  array to work on
      rep (float):  value to fill for invalid results

   Returns:
      sqrt(x):  array with same shape as x
   '''
   y = less_equal(x,0)
   y2 = isfinite(x)
   y = logical_and(y,y2)
   z = multiply(x,logical_not(y))
   return sqrt(z)+rep*y

def extrema(x):
   '''Returns the max/min of an array'''
   return asarray([minimum.reduce(x), maximum.reduce(x)])

def divz(x,y=1,repl=0.0,out=None):
   '''compute division with replacement of bad values.

   Args:
      x (array):  numerator
      y (array):  denominator
      repl (float): value to fill for invalid results
      out (bool):  output dtype of array.

   Returns:
      array:  result of x/y with replacemnt
   '''

   if len(shape(y)) or len(shape(x)):
      if len(shape(y)): bad = equal(y,0.0)
      else: bad = ones(x.shape)*(y==0.0)
      not_bad = 1-bad
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

def rms( x,y=None):
   '''Compute the RMS of an array

   Args:
      x (array): array to work on
      y (float):  the average of x

   Returns:
      float:  RMS
   '''
   if len(x) > 2:
      if y == None: y=average(x)
      return sqrt(mean(power(subtract(x,y),2)))
   else: return 0.0


def getexp(x,flo=-1000,fhi=1000):
   '''Compute the exp(x) with replacement to avoid overflows

   Args:
      x (array): array to work on
      flo (float):  floor value
      fhi (float):  ceiling value

   Returns:
      array:  exp(x) with values 0 were over/underflow occurs.
   '''

   x2 = x.astype(longfloat)
   y = greater(x2,flo)*less(x2,fhi)
   y2 = isfinite(x2)
   y = logical_and(y,y2)
   g = exp(y*x2)*y
   g = where(isnan(g),0.0,g).astype(x.dtype)
   return g

def bwt( x, iter=3):
   '''Copute the bi-weight estimator.

   Args:
      x (array):  array to work on
      iter (int):  number of iterations

   Returns:
      float:  the biweight estimator
   '''

   x=asarray(x)
   ns = sqrt(len(x))
   if len(x)>0: M = median(x)
   else: M=0.0
   if len(x) <= 2:
      S = 0*M
   else:
      MAD = median(abs(x-M))
      if not MAD.any(): MAD = rms(x-M)*0.6745
      if MAD.any():
         for i in range(iter):
            u6 = divz(x-M,MAD)/6.0
            omuu62 = power(1-power(u6,2),2)
            g6 = less(abs(u6),1.0)
            M = M + add.reduce(g6*(x-M)*omuu62)/add.reduce(g6*omuu62)
            u9 = divz(x-M,MAD)/9.0
            omuu9 = 1-power(u9,2)
            omuu59 = 1-5*power(u9,2)
            g9 = less(abs(u9),1.0)
            S = ns * divz(sqrt(add.reduce(g9*power(x-M,2)*power(omuu9,4))), 
                          abs(add.reduce(g9*omuu9*omuu59)))
            MAD = 0.6745*S
      else: M,S = mean(x),0.0
   return [M,S]

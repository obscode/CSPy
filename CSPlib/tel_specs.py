'''Module of telescope/instrument specs'''

data = {
      'SWO': {
         'NC': {
            'scale':0.435,
            'gain':1.040,
            'rnoise':3.4,
            'overscan':{
               1:[2070,2166],
            },
            'datasec':[1,2048,1,2056],
            'statsec':[300,1600,300,1600],
            'lincorr':{
               1:{
                  'c1':1.0,
                  'c2':-0.033,
                  'c3':0.0057,
                  'alpha':1.1150
                  },
               2:{
                  'c1':1.0,
                  'c2':-0.012,
                  'c3':0.0017,
                  'alpha':1.0128,
                  },
               3:{
                  'c1':1.0,
                  'c2':-0.010,
                  'c3':0.0014,
                  'alpha':1.0,
                  },
               4:{
                  'c1':1.0,
                  'c2':-0.014666,
                  'c3':0.0027,
                  'alpha':1.0696,
               },
            },
            'exposure':'@NEWEXPT',
            'filter':'@FILTER',
            'date':'@JD',
            'ncombine':'@NCOMBINE',
            'airmass':'@WAIRMASS',
            'object':'@OBJECT',
            'datamax':40000,
            'meansky':'@MEANSKY',
         },
      },
      'BAA': {
         'IM': {
            'scale':0.200,
            'gain':'@EGAIN',
            'rnoise':'@ENOISE',
            'overscan':{
               1:[2049,2112],
            },
            'datasec':[1,2048,1,4096],
            'statsec':[300,1600,300,3600],
            'exposure':'@EXPTIME',
            'filter':'@FILTER',
            'ncombine':'@NCOMBINE',
            'airmass':'@AIRMASS',
            'object':'@OBJECT',
            'datamax':65535,
         },
      },
      'SkyMapper': {
         'CCD': {
            'scale':0.4963,
            'gain':1.,
            'rnoise':0.0,
            'overscan':{
               1:[1300,1300],
            },
            'datasec':[1,1024,1,1024],
            'statsec':[0,1024,0,1024],
            'exposure':'@EXPTIME',
            'filter':'@FILTER',
            'date':'@JD',
            'ncombine':1,
            'airmass':'@WAIRMASS',
            'object':'@OBJECT',
         },
      },
      'PS': {
         'CCD': {
            'scale':0.25,
            'gain':1.,
            'rnoise':0.0,
            'overscan':{
               1:[1300,1300],
            },
            'datasec':[1,1024,1,1024],
            'statsec':[0,1024,0,1024],
            'exposure':'@EXPTIME',
            'filter':'@FILTER',
            'date':'@MJD-OBS',
            'ncombine':1,
            'airmass':'@AIRMASS',
            'object':'@OBJECT',
         },
      },
      'LCOGT': {
         'CCD': {
            'scale':'@PIXSCALE',
            'gain':'@GAIN',
            'rnoise':'@RDNOISE',
            'overscan':{
               1:None,
            },
            'datasec':[1,4096,1,4096],
            'statsec':[0,4096,0,4096],
            'exposure':'@EXPTIME',
            'filter':'@FILTER',
            'date':'@MJD-OBS',
            'ncombine':1,
            'airmass':'@AIRMASS',
            'object':'@OBJECT',
            'datamax':'@MAXLIN',
         },
      },
   }

def getTelIns(tel, ins):
   if tel not in data:
      raise ValueError('Telescope {} not found'.format(tel))
   if ins not in data[tel]:
      raise ValueError('Instrument {} not found for telescope {}'.format(
         tel, ins))

   return data[tel][ins]

def getTelInfo(key, hdr, tel, ins):
   '''Try to resolve a keyword from the telescope info and header. In order,
   we look for a value in the telescope specs, followed by a keyword "@KEY"
   lookup according to the telescope specs, and lastly just a keyword lookup
   in the header.'''

   inf = getTelIns(tel, ins)

   if key not in inf:
      '''Just grab the key from the header if it exists.'''
      if key not in hdr:
         raise AttributeError("key {} not in header or telescope info".format(
            key))
      return hdr[key]
   
   idx = inf[key]
   if type(idx) is str:
      if idx[0] == '@':
         if idx[1:] not in hdr:
            raise AttributeError("key {} not found in header".format(idx[1:]))
         return hdr[idx[1:]]

   return idx


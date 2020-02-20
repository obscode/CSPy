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
            'datasec':[0,2047,0,2055],
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



'''Module of telescope/instrument specs'''

data = {
      'SWO': {
         'NC': {
            'scale':0.435,
            'gain':1.040,
            'rnoise':3.4,
            }
         }
      }

def getTelIns(tel, ins):
   if tel not in data:
      raise ValueError('Telescope {} not found'.format(tel))
   if ins not in data[tel]:
      raise ValueError('Instrument {} not found for telescope {}'.format(
         tel, ins))

   return data[tel][ins]



'''Module for various tools to query ALERCE and associated thingies (like
TNS, etc).'''

import requests
import json
from astropy.table import Table
from astropy.time import Time
from CSPlib.config import getconfig
from collections import OrderedDict

cfg = getconfig()

filtDict = {1:'Zg', 2:'Zr'}     # convert filter id to filter name
TNSfiltDict = {111:'Zr', 110:'Zg',    # ZTF
               71: 'Ac', 72: 'Ao',    # ATLAS
               21: 'ASg',             # ASAS-SN
               75: 'GG',              # Gaia
               }
TNSGroups = {111: 'ZTF', 110:'ZTF',
             71: 'ATLAS', 72:'ATLAS',
             21: 'ASAS-SN',
             75: 'Gaia'}


def decodeTNSPhot(datum):
   # Discombobulate the OrderedDict output
   if 'jd' in datum:
      JD = datum['jd']
   elif 'obsdate' in datum:
      JD = Time(datum['obsdate']).jd
   else:
      return None
   if 'flux' in datum and datum['flux'] != '':
      mag = datum['flux']
      if datum['fluxerr'] == '':
         emag = 0
      else:
         emag = datum['fluxerr']
      upper = False
   elif 'limflux' in datum and datum['limflux'] != '':
      mag = datum['limflux']
      emag = -1
      upper = True
   if 'filters' in datum and datum['filters'] != '':
      if datum['filters']['id'] in TNSfiltDict:
         filt = TNSfiltDict[datum['filters']['id']]
         survey = TNSGroups[datum['filters']['id']]
      else:
         print("Warning: Uknown filter for:")
         print("filter:",datum['filters'])
         print("telescope:",datum['telescope'])
         filt = 'UN'
         survey = "unknown"
   return [JD, filt, mag, emag, upper, survey]

def getCandidatePhot(name):
   '''Given a candidate name, query ALERCE and/or TNS and get ZTF, ATLAS, etc
   photometry.'''

   url = 'http://api.alerce.online/ztf/v1/objects/{}/lightcurve'
   res = requests.get(url.format(name))
   if res.status_code != 200:
      # Search TNS
      url = 'https://www.wis-tns.org/api/get/search'
      tns_marker = 'tns_marker{"tns_id":129996,"type": "bot", '\
                   '"name":"POISE_bot"}'
      headers = {'User-Agent': tns_marker}
      searchobj = {"objname":name, "objname_exact_match":1}
      search_data = {'api_key':cfg.remote.TNS_API, 
                     'data':json.dumps(OrderedDict(searchobj))}
      response = requests.post(url, headers=headers, data=search_data)
      if response.status_code != 200:
         return None
      data = json.loads(response.text, object_pairs_hook=OrderedDict)['data']
      if len(data['reply']) == 0:
         return None
      
      # Now for the object
      url = 'https://www.wis-tns.org/api/get/object'
      del searchobj['objname_exact_match']
      searchobj['photometry'] = '1'
      searchobj['spectroscopy'] = '0'
      search_data['data'] = json.dumps(OrderedDict(searchobj))
      response = requests.post(url, headers=headers, data=search_data)
      if response.status_code != 200:
         return None
      data = json.loads(response.text,object_pairs_hook=OrderedDict)['data']['reply']
      rdata = [decodeTNSPhot(datum) for datum in data['photometry']]

   else:
      data = json.loads(res.content)
      rdata = []
      for item in data['detections']:
         rdata.append([item['mjd']+2400000.5, filtDict[item['fid']],
            item['magpsf'], item['sigmapsf'], False, 'ZTF'])

      for item in data['non_detections']:
         rdata.append([item['mjd']+2400000.5, filtDict[item['fid']],
            item['diffmaglim'], -1, True])

   return(Table(rows=rdata, 
         names=['JD','filt','mag','emag','upper','survey']))



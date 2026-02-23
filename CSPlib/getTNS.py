"""
TNS connectivity. Taken from the code on TNS examples.
original author:
@author: Nikola Knezevic ASTRO DATA
"""

import os
import requests
import json
import time
from collections import OrderedDict
from .config import getconfig


#--------------------- PARAMETERS -------------------#
tns = "www.wis-tns.org" # production
url_tns_api = "https://" + tns + "/api"

#tns_bot_id = "Here put your tns bot id."
#tns_bot_name = "Here put your tns bot name."
#tns_api_key = "Here put your tns api key."

# List that represents json file with all posible ("key", "value") parameters for searching obj on the tns
get_obj = [
           ("objname", ""), 
           ("objid", ""), 
           ("photometry", "0"), 
           ("spectra", "1")
          ]

ext_http_errors = [403, 500, 503]
err_msg = ["Forbidden", "Internal Server Error: Something is broken", "Service Unavailable"]
#-----------------------------------------------------------------------------------------------------------------#


#--------------------------------------------------- FUNCTIONS ---------------------------------------------------#
def set_bot_tns_marker(bot_id, bot_name):
    tns_marker = 'tns_marker{"tns_id": "' + str(bot_id) + '", "type": "bot", "name": "' + bot_name + '"}'
    return tns_marker

def format_to_json(source):
    parsed = json.loads(source, object_pairs_hook = OrderedDict)
    result = json.dumps(parsed, indent = 4)
    return result

def is_string_json(string):
    try:
        json_object = json.loads(string)
    except Exception:
        return False
    return json_object

def print_status_code(response):
    json_string = is_string_json(response.text)
    if json_string != False:
        print ("status code ---> [ " + str(json_string['id_code']) + " - '" + json_string['id_message'] + "' ]\n")
    else:
        status_code = response.status_code
        if status_code == 200:
            status_msg = 'OK'
        elif status_code in ext_http_errors:
            status_msg = err_msg[ext_http_errors.index(status_code)]
        else:
            status_msg = 'Undocumented error'
        print ("status code ---> [ " + str(status_code) + " - '" + status_msg + "' ]\n")

def getTNS(objname):
    cfg = getconfig()
    if cfg.remote.TNS_bot_id is None or cfg.remote.TNS_api_key is None \
        or cfg.remote.TNS_bot_name is None:
        raise RuntimeError("Error:  you have not setup your TNS credentials")

    get_obj_list = [("objname",objname),("photometry",1),("spectra",0)]

    get_url = url_tns_api + "/get/object"
    tns_marker = set_bot_tns_marker(cfg.remote.TNS_bot_id, cfg.remote.TNS_bot_name)
    headers = {'User-Agent': tns_marker}
    json_file = OrderedDict(get_obj_list)
    get_data = {'api_key': cfg.remote.TNS_api_key, 'data': json.dumps(json_file)}
    response = requests.post(get_url, headers = headers, data = get_data)

    json_data = format_to_json(response.text)
    d = json.loads(json_data)
    if d['id_code'] != 200:
        # failed
        return None
    return d['data']



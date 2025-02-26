# CSP
Repository for CSP-related code, scripts, etc.

Currently in this repository:

1. Python scripts
2. Photometry pipeline

To install the python-based software, simply do:
   
`pip install .`

## Note about Database access:

Since this is a public repository, the password to the CSP database is not
included in any of the code. When needed, you will be prompted for it, 
unless you set an environment variable:

`export CSPpasswd=XXXXXXXX`

or:

`setenv CSPpasswd XXXXXXXX`

## More detailed notes about intalling

The pipeline needs some extra stuff (source extractor, rclone) and some setup
to get it running correctly. I've made some notes:

https://docs.google.com/document/d/1zTJ4warFDNB5b4pACQBb_iv947kFq21iG6mb_xWrwdg/edit?usp=sharing

# CSP
Repository for CSP-related code, scripts, etc.

Currently in this repository:

1. Python scripts
2. Photometry pipeline (soon)

To install the python-based software, simply do:
   
`python setup.py install`

## Note about Database access:

Since this is a public repository, the password to the CSP database is not
included in any of the code. When needed, you will be prompted for it, 
unless you set an environment variable:

`export CSPpasswd=XXXXXXXX`

or:

`setenv CSPpasswd XXXXXXXX`

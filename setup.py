from distutils.core import setup
from glob import glob
import os
scripts = glob('bin/*')

#files = glob('data/*')
#dirs = [f for f in files if os.path.isdir(f)]
#files = [f for f in files if os.path.isfile(f)]
#data_files = [('data',files)]
#for d in dirs:
#   data_files.append((d, glob(os.path.join(d,'*'))))

setup(
      name='CSPlib',
      version='0.1.0',
      author='Chris Burns',
      author_email='cburns@carnegiescience.edu',
      packages=['CSPlib'],
      scripts=scripts,
      package_data={'CSPlib':['data/*','data/*/*']},
      #data_files=data_files,
      description='CSP-related python code and scripts',
      requires=[
         'astropy',
         'pymysql',
         'numpy',
      ],
      )


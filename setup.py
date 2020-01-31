from distutils.core import setup
from glob import glob
scripts = glob('bin/*')

setup(
      name='CSPlib',
      version='0.1.0',
      author='Chris Burns',
      author_email='cburns@carnegiescience.edu',
      packages=['CSPlib'],
      scripts=scripts,
      package_data={'CSPlib':['data/*']},
      description='CSP-related python code and scripts',
      requires=[
         'astropy',
         'pymysql',
         'numpy',
      ],
      )


from distutils.core import setup

setup(
      name='CSPlib',
      version='0.1.0',
      author='Chris Burns',
      author_email='cburns@carnegiescience.edu',
      packages=['CSPlib'],
      scripts=['bin/make_optls_table.py','bin/make_nirls_table.py'],
      include_package_data=True,
      description='CSP-related python code and scripts',
      install_requires=[
         'astropy',
         'pymysql',
         'numpy',
         'scipy',
         'matplotlib',
      ],
      )


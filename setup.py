from distutils.core import setup

setup(
      name='CSPlib',
      version='0.1.0',
      author='Chris Burns',
      author_email='cburns@carnegiescience.edu',
      packages=['CSPlib'],
      scripts=['bin/make_optls_table','bin/make_nirls_table'],
      package_data={'CSPlib':['data/*']},
      description='CSP-related python code and scripts',
      requires=[
         'astropy',
         'pymysql',
         'numpy',
      ],
      )


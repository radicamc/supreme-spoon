#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='supreme-spoon',
      version='0.0.1',
      license='MIT',
      author='Michael Radica',
      author_email='michael.radica@umontreal.ca',
      packages=['supreme-spoon'],
      include_package_data=True,
      url='https://github.com/radicamc/supreme-spoon',
      description='Tools for Reduction of NIRISS/SOSS TSOs',
      package_data={'': ['README.md', 'LICENSE']},
      install_requires=['applesoss', 'astropy', 'jwst', 'matplotlib',
                        'numpy', 'scipy', 'tqdm'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        ],
      )

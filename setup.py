#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='supreme_spoon',
      version='0.2.0',
      license='MIT',
      author='Michael Radica',
      author_email='michael.radica@umontreal.ca',
      packages=['supreme_spoon'],
      include_package_data=True,
      url='https://github.com/radicamc/supreme-spoon',
      description='Tools for Reduction of NIRISS/SOSS TSOs',
      package_data={'': ['README.md', 'LICENSE']},
      install_requires=['applesoss', 'astropy', 'bottleneck', 'corner',
                        'exotic_ld', 'juliet', 'jwst', 'matplotlib', 'numpy',
                        'pandas', 'ray', 'scipy', 'tqdm', 'pyyaml'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        ],
      )

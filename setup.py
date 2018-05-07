from setuptools import setup
import os

print("""
*******************************************************************
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
(c) 2017 Haotian Teng
*******************************************************************
""")

install_requires=[
  'h5py>=2.7.0',
  'mappy>=2.10.0',
  'numpy>=1.13.3',
  'statsmodels>=0.8.0',
  'tqdm>=4.23.0'
]
extras_require={
  "tf": ["tensorflow>=1.3.0"],
  "tf_gpu": ["tensorflow-gpu>=1.3.0"],
}

setup(
  name = 'chiron',
  packages = ['chiron'], # this must be the same as the name above
  version = '0.4',
  include_package_data=True,
  description = 'A deep neural network basecaller for nanopore sequencing.',
  author = 'Haotian Teng',
  author_email = 'havens.teng@gmail.com',
  url = 'https://github.com/haotianteng/chiron', 
  download_url = 'https://github.com/haotianteng/chiron/archive/0.4.tar.gz', # I'll explain this in a second
  keywords = ['basecaller', 'nanopore', 'sequencing','neural network'], # arbitrary keywords
  license="MPL 2.0",
  classifiers = ['License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)'],
  install_requires=install_requires,
  extras_require=extras_require,
  entry_points={'console_scripts':['chiron=chiron.entry:main'],}
)

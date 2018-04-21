from setuptools import setup

print("""
*******************************************************************
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
(c) 2017 Haotian Teng
*******************************************************************
""")

setup(
  name = 'chiron',
  packages = ['chiron'], # this must be the same as the name above
  version = '0.3',
  include_package_data=True,
  description = 'A deep neural network basecaller for nanopore sequencing.',
  author = 'Haotian Teng',
  author_email = 'havens.teng@gmail.com',
  url = 'https://github.com/haotianteng/chiron', 
  download_url = 'https://github.com/haotianteng/chiron/archive/0.3.tar.gz', # I'll explain this in a second
  keywords = ['basecaller', 'nanopore', 'sequencing','neural network'], # arbitrary keywords
  license="MPL 2.0",
  classifiers = ['License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)'],
  install_requires=['tensorflow>=1.5.0','h5py>=2.7.0', 'numpy>=1.13.3', 'statsmodels>=0.8.0', 'tqdm>=4.23.0', 'mappy>=2.10.0'],
  entry_points={'console_scripts':['chiron=chiron.entry:main'],}
)

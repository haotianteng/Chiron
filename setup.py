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
  'tqdm>=4.23.0',
  'scipy>=1.2.1',
  'biopython>=1.73'
]
extras_require={
  "tf": ["tensorflow>=1.13.1"],
  "tf_gpu": ["tensorflow-gpu>=1.13.1"],
}
exec(open('chiron/_version.py').read())
setup(
  name = 'chiron',
  packages = ['chiron'], 
  version = __version__,
  include_package_data=True,
  description = 'A deep neural network basecaller for nanopore sequencing.',
  author = 'Haotian Teng',
  author_email = 'havens.teng@gmail.com',
  url = 'https://github.com/haotianteng/chiron', 
  download_url = 'https://github.com/haotianteng/Chiron/archive/0.5.tar.gz', 
  keywords = ['basecaller', 'nanopore', 'sequencing','neural network'], 
  license="MPL 2.0",
  classifiers = ['License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)'],
  install_requires=install_requires,
  extras_require=extras_require,
  entry_points={'console_scripts':['chiron=chiron.entry:main'],}
)

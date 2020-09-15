from setuptools import setup, find_packages
from glob import glob

setup(
   name='miri3d',
   version='1.0.0',
   description='MIRI 3d cube tools',
   author='David R. Law',
   author_email='dlaw@stsci.edu',
   packages=find_packages(),
   data_files=[
       ('data/lvl2btemplate',glob('data/lvl2btemplate/*'))
   ],
   include_package_data=True,
   install_requires=['miricoord'],
)

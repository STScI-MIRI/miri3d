from setuptools import setup, find_packages

setup(
   name='miri3d',
   version='1.0.0',
   description='MIRI 3d cube tools',
   author='David R. Law',
   author_email='dlaw@stsci.edu',
   packages=find_packages(),
   install_requires=['miricoord'],
)

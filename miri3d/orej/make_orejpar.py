#
"""
Python tools for creating the MIRI MRS outlier rejection parameter files.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
26-May-2020  First written (D. Law)

"""

from astropy.io import fits
from astropy.time import Time
import datetime
import os as os
import numpy as np
import pdb

#############################

# This routine is the master function to make the reference file
def make_orejpar():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'orejpar/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-orejpar-'+str(mjd)+'.fits'
    outfile=os.path.join(outdir,filename)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # Create primary hdu (blank data with header)
    print('Making 0th extension')
    hdu0=make_ext0(now,filename)

    # Create first extension (OREJPAR: rejection threshholds, etc)
    print('Making 1st extension')
    hdu1=make_ext1()
    
    hdul=fits.HDUList([hdu0,hdu1])
    hdul.writeto(outfile,overwrite=True)

#############################

# Create blank primary extension and header with base instrument information

def make_ext0(now,thisfile):
    hdu=fits.PrimaryHDU()
    
    hdu.header['DATE']=now.value

    hdu.header['REFTYPE']='OREJPAR'
    hdu.header['DESCRIP']='Default IFU Cube outlier rejection parameters'
    hdu.header['PEDIGREE']='GROUND'
    hdu.header['TELESCOP']='JWST'
    hdu.header['INSTRUME']='MIRI'
    hdu.header['MODELNAM']='FM'
    hdu.header['DETECTOR']='N/A'
    hdu.header['EXP_TYPE']='MIR_MRS'
    hdu.header['BAND']='N/A'
    hdu.header['CHANNEL']='N/A'

    hdu.header['FILENAME']=thisfile
    hdu.header['USEAFTER']='2000-01-01T00:00:00'
    hdu.header['VERSION']=int(now.mjd)
    hdu.header['AUTHOR']='D. Law'
    hdu.header['ORIGIN']='STSCI'
    hdu.header['HISTORY']='IFU Cube outlier rejection defaults'
    hdu.header['HISTORY']='DOCUMENT: TBD'
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/orej/make_orejpar.py'
    hdu.header['HISTORY']='DATA USED: Simulated data created by D. Law'

    return hdu

#############################

# Create 1st extension (OREJPAR) with basic information

def make_ext1():    
    chan=np.array([1,1,1,2,2,2,3,3,3,4,4,4])
    bnd=np.array(['SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG'])

    nlow=np.zeros(12) # Inherited from default
    nhigh=np.zeros(12) # Inherited from default
    maskpt=np.ones(12)*0.7 # Inherited from default
    grow=np.ones(12) # Inherited from default
    snr1=np.ones(12)*4. # Inherited from default
    snr2=np.ones(12)*3. # Inherited from default
    scale1=np.ones(12)*2.4 # Set based on initial outlier rejection tests
    scale2=np.ones(12)*2.4 # Set based on initial outlier rejection tests

    col1=fits.Column(name='CHANNEL',format='I',array=chan)
    col2=fits.Column(name='BAND',format='10A',array=bnd)
    col3=fits.Column(name='NLOW',format='I',array=nlow)
    col4=fits.Column(name='NHIGH',format='I',array=nhigh)
    col5=fits.Column(name='MASKPT',format='E',array=maskpt)
    col6=fits.Column(name='GROW',format='I',array=grow)
    col7=fits.Column(name='SNR1',format='E',array=snr1)
    col8=fits.Column(name='SNR2',format='E',array=snr2)
    col9=fits.Column(name='SCALE1',format='E',array=scale1)
    col10=fits.Column(name='SCALE2',format='E',array=scale2)    

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10])
    hdu.header['EXTNAME']='OREJPAR'
    
    return hdu

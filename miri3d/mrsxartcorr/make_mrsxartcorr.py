#
"""
Python tools for creating the MIRI MRS cross-artifact
correction reference files.

Model consists of a wide lorentzian plus 4 gaussians with many tied parameters
Don't store polynomials, store vectors of values up y
This gives flexibility in case we need more polynomial terms, or if some part can't be described
by a polynomial

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
07-Mar-2022  First written (D. Law)
20-Apr-2022  Updated crds type name (D. Law)
"""

from astropy.io import fits
from astropy.time import Time
from astropy.table import vstack, Table
import datetime
import os as os
import numpy as np
import miricoord.mrs.mrs_tools as mt
import pdb
from jwst import datamodels

def make_mrsxartcorr():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'mrsxartcorr/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-mrsxartcorr-'+str(mjd)+'.fits'
    outfile=os.path.join(outdir,filename)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)
    
    # Create primary hdu (blank data with header)
    print('Making 0th extension')
    hdu0=make_ext0(now,filename)

    # Create extensions
    print('Making extensions')
    hdu1A=read_ext('reftype_chx.fits','CH1A')
    hdu1B=read_ext('reftype_chx.fits','CH1B')
    hdu1C=read_ext('reftype_chx.fits','CH1C')
    
    hdul=fits.HDUList([hdu0,hdu1A,hdu1B,hdu1C])
    hdul.writeto(outfile,overwrite=True)

    print('Wrote output file to ',outfile)

    # Test that it passes the datamodel
    with datamodels.open(outfile) as im:
        assert isinstance(im, datamodels.MirMrsXArtCorrModel)
    
#############################

# Create blank primary extension and header with base instrument information

def make_ext0(now,thisfile):
    hdu=fits.PrimaryHDU()
    
    hdu.header['DATE']=now.value

    hdu.header['REFTYPE']='MRSXARTCORR'
    hdu.header['DESCRIP']='MRS Cross Artifact Correction'
    hdu.header['PEDIGREE']='GROUND'
    hdu.header['DATAMODL']='MirMrsXArtCorrModel'
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
    hdu.header['HISTORY']='DOCUMENT: TBD'
    hdu.header['HISTORY']='SOFTWARE: '

    return hdu

#############################

# Read extension from a file

def read_ext(file,extname):
    hdu = fits.open(file)

    hdu1 = hdu[1]
    hdu1.header['EXTNAME'] = extname

    return hdu1

#############################

# Create 1A extension with cross-artifact model information

def make_ext1A():
    yrow = np.arange(1024)

    # Read vectors from where????
#    lfwhm = readsomethinghere

    col1=fits.Column(name='YROW',format='I',array=yrow, unit='pixel')
    col2=fits.Column(name='LOR_FWHM',format='E',array=lfwhm, unit='pixel')
    col3=fits.Column(name='LOR_SCALE',format='E',array=lscale)
    col4=fits.Column(name='GAU_FWHM',format='E',array=gfwhm, unit='pixel')
    col5=fits.Column(name='GAU_XOFF',format='E',array=gxoff, unit='pixel')
    col6=fits.Column(name='GAU_SCALE1',format='E',array=gscale1, unit='pixel')
    col7=fits.Column(name='GAU_SCALE2',format='E',array=gscale2, unit='pixel')
    
    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7])
    hdu.header['EXTNAME']='CH1A'

    return hdu

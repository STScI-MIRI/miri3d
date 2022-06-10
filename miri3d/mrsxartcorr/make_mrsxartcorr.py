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
01-Jun-2022  Update with flight data (D. Law)
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

    # Create extensions from full model
    print('Making extensions')
    hdu1A=read_ext('xart_1A_model.fits','CH1A')
    hdu1B=read_ext('xart_1B_model.fits','CH1B')
    hdu1C=read_ext('xart_1C_model.fits','CH1C')
    hdu2A=read_ext('xart_2A_model.fits','CH2A')
    hdu2B=read_ext('xart_2B_model.fits','CH2B')
    hdu2C=read_ext('xart_2C_model.fits','CH2C')
    
    # Create placeholder extensions with zero models
    hdu3A=zero_ext(hdu1A,'Ch3A')
    hdu3B=zero_ext(hdu1A,'Ch3B')
    hdu3C=zero_ext(hdu1A,'Ch3C')   
    hdu4A=zero_ext(hdu1A,'Ch4A')
    hdu4B=zero_ext(hdu1A,'Ch4B')
    hdu4C=zero_ext(hdu1A,'Ch4C')    
    
    hdul=fits.HDUList([hdu0,hdu1A,hdu1B,hdu1C,hdu2A,hdu2B,hdu2C,hdu3A,hdu3B,hdu3C,hdu4A,hdu4B,hdu4C])
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
    hdu.header['PEDIGREE']='INFLIGHT 2022-05-22 2022-05-22'
    hdu.header['DATAMODL']='MirMrsXArtCorrModel'
    hdu.header['TELESCOP']='JWST'
    hdu.header['INSTRUME']='MIRI'
    hdu.header['MODELNAM']='FM'
    hdu.header['DETECTOR']='N/A'
    hdu.header['EXP_TYPE']='MIR_MRS'
    hdu.header['BAND']='N/A'
    hdu.header['CHANNEL']='N/A'

    hdu.header['FILENAME']=thisfile
    hdu.header['USEAFTER']='2022-05-01T00:00:00'
    hdu.header['VERSION']=int(now.mjd)
    hdu.header['AUTHOR']='D. Law'
    hdu.header['ORIGIN']='STSCI'
    hdu.header['HISTORY']='Initial flight models'
    hdu.header['HISTORY']='DOCUMENT: TBD'
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/blob/master/miri3d/mrsxartcorr/make_mrsxartcorr.py'

    return hdu

#############################

# Read extension from a file

def read_ext(file,extname):
    hdu = fits.open(file)

    hdu1 = hdu[1]
    hdu1.header['EXTNAME'] = extname

    return hdu1

#############################

# Create a zero-valued placeholder extension
# based on an input structure

def zero_ext(template,extname):
    hdu = template.copy()
    hdu.header['EXTNAME'] = extname

    hdu.data['LOR_FWHM'][:] = 100.
    hdu.data['LOR_SCALE'][:] = 0.
    hdu.data['GAU_FWHM'][:] = 1.
    hdu.data['GAU_XOFF'][:] = 10.
    hdu.data['GAU_SCALE1'][:] = 0.
    hdu.data['GAU_SCALE2'][:] = 0.
    
    return hdu

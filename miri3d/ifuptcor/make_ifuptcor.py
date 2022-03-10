#
"""
Python tools for creating the MIRI MRS detector-based point source
spectral extraction correction files.  These roll up the
spectral leak correction, across-slice wavelength correction,
and across-slice transmission correction.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
24-Feb-2022  First written (D. Law)
"""

from astropy.io import fits
from astropy.time import Time
from astropy.table import vstack, Table
import datetime
import os as os
import numpy as np
import miricoord.mrs.mrs_tools as mt
import pdb

def make_ifuptcor():
    # Input files
    leakfile='yannis_order2.fits'
    tracor12='MIRI_FM_MIRIFUSHORT_12_TRACORR_06.00.00.fits'
    tracor34='MIRI_FM_MIRIFULONG_34_TRACORR_06.00.00.fits'
    wavcor12='MIRI_FM_MIRIFUSHORT_12_WAVCORR_06.00.00.fits'
    wavcor34='MIRI_FM_MIRIFULONG_34_WAVCORR_06.00.00.fits'
    
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'ifuptcor/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-ifuptcor-'+str(mjd)+'.fits'
    outfile=os.path.join(outdir,filename)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)
    
    # Create primary hdu (blank data with header)
    print('Making 0th extension')
    hdu0=make_ext0(now,filename)

    # Create first extension (Spectral Leak)
    print('Making 1st extension')
    hdu1a,hdu1b,hdu1c,hdu1d=make_ext1(leakfile)
 
    # Create second extension (Across-slice Transmission)
    print('Making 2nd extension')
    hdu2=make_ext2(tracor12,tracor34)

    # Create third extension (Across-slice Wavelength)
    print('Making 3rd extension')
    hdu3a,hdu3b,hdu3c=make_ext3(wavcor12,wavcor34)

    hdul=fits.HDUList([hdu0,hdu1a,hdu1b,hdu1c,hdu1d,hdu2,hdu3a,hdu3b,hdu3c])
    hdul.writeto(outfile,overwrite=True)

    print('Wrote output file to ',outfile)
    
#############################

# Create blank primary extension and header with base instrument information

def make_ext0(now,thisfile):
    hdu=fits.PrimaryHDU()
    
    hdu.header['DATE']=now.value

    hdu.header['REFTYPE']='IFUPTCOR'
    hdu.header['DESCRIP']='IFU Point Source Corrections'
    hdu.header['PEDIGREE']='GROUND'
    hdu.header['DATAMODL']='MiriIFUPtCorrModel'
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
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/ifuptcor/make_ifuptcor.py'

    return hdu

#############################

# Create 1st extension with spectral leak correction information

def make_ext1(leakfile):
    hdu=fits.open(leakfile)

    hdu1a=hdu['SCI']
    hdu1b=hdu['ERR']
    hdu1c=hdu['DQ']
    hdu1d=hdu['DQ_DEF']

    return hdu1a, hdu1b, hdu1c, hdu1d

#############################

# Create extensions with across-slice transmission correction

def make_ext2(tracor12,tracor34):
    t12 = Table.read(tracor12)
    t34 = Table.read(tracor34)
    t1234 = vstack([t12,t34])

    hdu=fits.BinTableHDU(t1234)
    hdu.header['EXTNAME']='TRACOR'

    return hdu

#############################

# Create extensions with across-slice wavelength correction

def make_ext3(wavcor12,wavcor34):
    # The two FITS files actually have identical contents, so we
    # just need to read one of them
    
    t12a = Table.read(wavcor12,hdu='WAVCORR_OPTICAL')
    hdu12a = fits.BinTableHDU(t12a)
    hdu12a.header['EXTNAME']='WAVCORR_OPTICAL'

    t12b = Table.read(wavcor12,hdu='WAVCORR_XSLICE')
    hdu12b = fits.BinTableHDU(t12b)
    hdu12b.header['EXTNAME']='WAVCORR_XSLICE'

    t12c = Table.read(wavcor12,hdu='WAVCORR_SHIFT')
    hdu12c = fits.BinTableHDU(t12c)
    hdu12c.header['EXTNAME']='WAVCORR_SHIFT'
    
    return hdu12a, hdu12b, hdu12c


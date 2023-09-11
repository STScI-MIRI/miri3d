#
"""
Python tools for creating the MIRI MRS detector-based point source
spectral extraction correction files.  These roll up the
spectral leak correction, across-slice wavelength correction,
and across-slice transmission correction.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
24-Feb-2022  First written (D. Law)
20-Apr-2022  Updated data model names (D. Law)
17-Aug-2023  Update leak correction format (D. Law)
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
import pydl.pydlutils.bspline as bs
import matplotlib.pyplot as plt

# Bspline iteration doesn't seem to be working, write my own in a wrapper
def bspline_wrap(xvec,yvec,nbkpts=50,wrapsig_low=3,wrapsig_high=3,wrapiter=10,verbose=False):
    xvec_use=xvec.copy()
    yvec_use=yvec.copy()
    
    for ii in range(0,wrapiter):
        sset,_=bs.iterfit(xvec_use,yvec_use,nbkpts=nbkpts)
        ytemp,_=sset.value(xvec_use)
        diff=yvec_use-ytemp
        rms=np.nanstd(diff)
        rej=np.where((diff < -wrapsig_low*rms)|(diff > wrapsig_high*rms))
        keep=np.where((diff >= -wrapsig_low*rms)&(diff <= wrapsig_high*rms))
        if verbose:
            print('Rejected',len(rej[0]),'Kept',len(keep[0]))
        xvec_use=xvec_use[keep]
        yvec_use=yvec_use[keep]
        
    return sset

def make_mrsptcorr():
    # Input files
    leakfile='MIRI_FM_MIRIFULONG_34SHORT_SECONDORDER_PHOTOM_08.04.00.fits'
    tracor12='MIRI_FM_MIRIFUSHORT_12_TRACORR_06.00.00.fits'
    tracor34='MIRI_FM_MIRIFULONG_34_TRACORR_06.00.00.fits'
    wavcor12='MIRI_FM_MIRIFUSHORT_12_WAVCORR_06.00.00.fits'
    wavcor34='MIRI_FM_MIRIFULONG_34_WAVCORR_06.00.00.fits'

    # Ch3A PHOTOM file
    photom3a='MIRI_FM_MIRIFULONG_34SHORT_PHOTOM_aug2023.fits'
    
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'mrsptcorr/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-mrsptcorr-'+str(mjd)+'.fits'
    imgname='leak-'+str(mjd)+'.png'
    outfile=os.path.join(outdir,filename)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)
    outimg=os.path.join(outdir,imgname)
    
    # Create primary hdu (blank data with header)
    print('Making 0th extension')
    hdu0=make_ext0(now,filename)

    # Create first extension (Spectral Leak)
    print('Making 1st extension')
    hdu1=make_ext1(leakfile,photom3a,outimg)
 
    # Create second extension (Across-slice Transmission)
    print('Making 2nd extension')
    hdu2=make_ext2(tracor12,tracor34)

    # Create third extension (Across-slice Wavelength)
    print('Making 3rd extension')
    hdu3a,hdu3b,hdu3c=make_ext3(wavcor12,wavcor34)

    hdul=fits.HDUList([hdu0,hdu1,hdu2,hdu3a,hdu3b,hdu3c])
    hdul.writeto(outfile,overwrite=True)

    print('Wrote output file to ',outfile)

    # Test that it passes the datamodel
    # Note that this doesn't seem to be working right
    with datamodels.open(outfile) as im:
        assert isinstance(im, datamodels.MirMrsPtCorrModel)
    
#############################

# Create blank primary extension and header with base instrument information

def make_ext0(now,thisfile):
    hdu=fits.PrimaryHDU()
    
    hdu.header['DATE']=now.value

    hdu.header['REFTYPE']='MRSPTCORR'
    hdu.header['DESCRIP']='MRS Point Source Corrections'
    hdu.header['PEDIGREE']='INFLIGHT 2022-07-23 2022-07-23'
    hdu.header['DATAMODL']='MirMrsPtCorrModel'
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
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/mrsptcorr/make_mrsptcorr.py'
    hdu.header['HISTORY']='Updated format Sep 2023'

    return hdu

#############################

# Create 1st extension with spectral leak correction information

def make_ext1(leakfile,photom3a,outimg):
    # Open the CDP file
    hdu=fits.open(leakfile)
    leak=hdu['SCI'].data
    hdu.close()
    
    # Map of Ch3A wavelengths
    waveim=mt.waveimage('3A')
    wave1d=np.arange(11.55,13.47,0.0015) # Ch3A wavelengths, doesn't have to be exact

    # Distill 2d leak information from CDP file to 1d
    indx=np.argsort(waveim.ravel())
    temp1=waveim.ravel()[indx]
    temp2=leak.ravel()[indx]
    leak1d=np.interp(wave1d,temp1,temp2)
    
    # Bspline fit the 1d spectral leak response
    # (fit in the inverse to avoid huge-number issues at ends of the range)
    xtemp=wave1d
    ytemp=1/leak1d
    indx=np.where(np.isfinite(ytemp) == True)
    sset=bspline_wrap(xtemp[indx],ytemp[indx],nbkpts=80,wrapsig_low=2.5,wrapsig_high=2.5,wrapiter=10,verbose=False)
    leak1d_fit,_=sset.value(wave1d)
    leak1d_fit=1/leak1d_fit
    # Now in units of (DN/s) / (MJy/sr)
    # I.e., for a given Ch1B flux in MJy/sr, what DN/s does it produce in 3A?

    # Now read in the Ch3A flux calibration vector to give a fractional response,
    # For a given Ch1B flux in MJy/sr, what is the Ch3A flux in MJy/sr?
    photom3a_vals=(fits.open(photom3a))['SCI'].data
    # Distill photom file into a 1d vector
    indx=np.argsort(waveim.ravel())
    temp1=waveim.ravel()[indx]
    temp2=photom3a_vals.ravel()[indx]
    photom3a_1d=np.interp(wave1d,temp1,temp2)

    # Compute the ratio: multiple this by Ch1B flux to get the Ch3A leak flux
    ratio_noisy = photom3a_1d/leak1d
    ratio_fit = photom3a_1d/leak1d_fit

    plt.plot(wave1d,ratio_noisy)
    plt.plot(wave1d,ratio_fit)
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('Fractional leak')
    plt.savefig(outimg)

    # Placeholder error vector for future use
    ratio_err=np.zeros_like(ratio_fit)

    # Set up the FITS table
    col1=fits.Column(name='WAVELENGTH',format='E',array=wave1d,unit='micron')
    col2=fits.Column(name='FRAC_LEAK',format='E',array=ratio_fit)
    col3=fits.Column(name='ERR_LEAK',format='E',array=ratio_err)
    
    hdu=fits.BinTableHDU.from_columns([col1,col2,col3])
    hdu.header['EXTNAME']='LEAKCOR'

    return hdu

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


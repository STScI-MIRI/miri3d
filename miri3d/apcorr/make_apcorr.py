#
"""
Python tools for creating the MIRI MRS 1d extraction reference files.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
12-Feb-2020  First written (D. Law)
"""

from astropy.io import fits
from astropy.time import Time
import datetime
import os as os
import numpy as np
import pdb
from matplotlib import pyplot as plt
import miri3d.cubepar.make_cubepar as mc

#############################

def make_apcorrpar():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'apcorr/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-apcorrpar-'+str(mjd)+'.fits'
    outfile=os.path.join(outdir,filename)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # CDP input directory
    cdp_dir=os.path.join(data_dir,'apcorr/cdp/')
    
    # Create primary hdu (blank data with header)
    print('Making 0th extension')
    hdu0=make_ext0(now,filename)

    # Create first extension (APCORR)
    print('Making 1st extension')
    hdu1=make_ext1(cdp_dir)

    hdul=fits.HDUList([hdu0,hdu1])
    hdul.writeto(outfile,overwrite=True)
    
#############################

def make_ext0(now,thisfile):
    hdu=fits.PrimaryHDU()
    
    hdu.header['DATE']=now.value

    hdu.header['REFTYPE']='APCORR'
    hdu.header['DESCRIP']='Default MIRI MRS Aperture correction parameters'
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
    hdu.header['HISTORY']='IFU Cube defaults'
    hdu.header['HISTORY']='DOCUMENT: TBD'
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/apcorr/make_apcorr.py'
    hdu.header['HISTORY']='DATA USED: CDP-7'
    return hdu
    
#############################

def make_ext1(cdp_dir):
    print('Figuring out wavelength ranges')
    wmin1A,_=mc.waveminmax('1A')
    _,wmax4C=mc.waveminmax('4C')

    print('Building tables')

    # Set up placeholder vectors
    waves=np.arange(wmin1A,wmax4C,0.01)
    nwave=len(waves)
    radius=np.ones(nwave)
    inbkg=np.zeros(nwave)
    outbkg=np.zeros(nwave)
    axratio=np.ones(nwave)
    axangle=np.zeros(nwave)
    apcor=np.ones(nwave)
    aperr=np.zeros(nwave)

    # Populate real values
    # Read in the CDP files
    files=['MIRI_FM_MIRIFUSHORT_1SHORT_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFUSHORT_1MEDIUM_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFUSHORT_1LONG_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFUSHORT_2SHORT_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFUSHORT_2MEDIUM_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFUSHORT_2LONG_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFULONG_3SHORT_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFULONG_3MEDIUM_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFULONG_3LONG_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFULONG_4SHORT_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFULONG_4MEDIUM_APERCORR_07.00.00.fits',
           'MIRI_FM_MIRIFULONG_4LONG_APERCORR_07.00.00.fits']
    inwave=[]
    inap=[]
    incor=[]
    for file in files:
        hdu=fits.open(os.path.join(cdp_dir,file))
        data=hdu[1].data
        inwave.append(data['wavelength'])
        inap.append(data['a_aperture'])
        incor.append(data['aper_corr'])

    # Compile into big vectors
    # Simple polynomial fit to the aperture
    thefit=np.polyfit(np.array(inwave).ravel(),np.array(inap).ravel(),1)
    poly=np.poly1d(thefit)
    radius=poly(waves)
    
    # At present the CDP aperture-correction factors have unphysical features
    # set it equal to the median value
    apcor[:]=np.median(incor)

    #plt.plot(inwave,inap,'.')
    #plt.plot(waves,fitap)

    col1=fits.Column(name='WAVELENGTH',format='E',array=waves, unit='micron')
    col2=fits.Column(name='RADIUS',format='E',array=radius, unit='arcsec')
    col3=fits.Column(name='INNER_BKG',format='E',array=inbkg, unit='arcsec')
    col4=fits.Column(name='OUTER_BKG',format='E',array=outbkg, unit='arcsec')
    col5=fits.Column(name='AXIS_RATIO',format='E',array=axratio)
    col6=fits.Column(name='AXIS_PA',format='E',array=axangle, unit='degrees')
    col7=fits.Column(name='APCORR',format='E',array=apcor)
    col8=fits.Column(name='APCORR_ERR',format='E',array=aperr)

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7,col8])
    hdu.header['EXTNAME']='APCORR'
    
    return hdu

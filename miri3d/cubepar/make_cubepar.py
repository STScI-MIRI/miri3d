#
"""
Python tools for creating the MIRI MRS cubepar parameter files.
These define the values of things like spaxel size, wavelength solution,
and weight function parameters for the JWST pipeline cube building algorithm.

Author: Beth Sargent (sargent@stsci.edu) and David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
Mid-2018  IDL version written by Beth Sargent (sargent@stsci.edu)
18-Jul-2019  Convert to python (D. Law)
19-Jul-2019  Add exponential weight extensions (D. Law)
"""

from astropy.io import fits
from astropy.time import Time
import datetime
import os as os
import numpy as np
import pdb

#############################

# This routine is the master function to make the reference file
def make_cubepar():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'cubepar/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-cubepar-'+str(mjd)+'.fits'
    outfile=os.path.join(outdir,filename)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)
    
    # Create primary hdu (blank data with header)
    hdu0=make_ext0(now,filename)

    # Create first extension (CUBEPAR: basic spaxel size, etc)
    hdu1=make_ext1()

    # Create second extension (CUBEPAR_MSM: per-band 1/r2 weights and roi)
    hdu2=make_ext2()

    # Create third extension (MULTICHANNEL_MSM: multichannel 1/r2 weights and roi)
    hdu3=make_ext3()

    # Create the fourth extension (CUBEPAR_EMSM: per-band exponential weights and roi)
    hdu4=make_ext4()
    
    # Create the fifth extension (MULTICHANNEL_EMSM: multichannel exponential weights and roi)
    hdu5=make_ext5()
    
    hdul=fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4,hdu5])
    hdul.writeto(outfile,overwrite=True)

#############################

# Create blank primary extension and header with base instrument information

def make_ext0(now,thisfile):
    hdu=fits.PrimaryHDU()
    
    hdu.header['DATE']=now.value

    hdu.header['REFTYPE']='CUBEPAR'
    hdu.header['DESCRIP']='Default IFU Cube Sampling and weight parameters'
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
    hdu.header['AUTHOR']='B. Sargent and D. Law'
    hdu.header['ORIGIN']='STSCI'
    hdu.header['HISTORY']='IFU Cube defaults'
    hdu.header['HISTORY']='DOCUMENT: TBD'
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/cubepar/make_cubepar.py'
    hdu.header['HISTORY']='DATA USED: Simulated data created by B. Sargent and D. Law'
    hdu.header['HISTORY']='DIFFERENCES: Migrated to python creation code.'
    hdu.header['HISTORY']='DIFFERENCES: Add support for e^(-r) weight function.'   

    return hdu

#############################

# Create 1st extension (CUBEPAR) with basic cube setup information

def make_ext1():    
    chan=np.array([1,1,1,2,2,2,3,3,3,4,4,4])
    bnd=np.array(['SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG'])
    wmin=np.array([4.89,5.65,6.52,7.49,8.65,9.99,11.53,13.37,15.44,17.66,20.54,23.95])
    wmax=np.array([5.75,6.64,7.66,8.78,10.14,11.71,13.48,15.63,18.05,20.92,24.40,28.45])
    spaxsize=np.array([0.13,0.13,0.13,0.17,0.17,0.17,0.20,0.20,0.20,0.35,0.35,0.35])
    wsamp=np.array([0.001,0.001,0.001,0.002,0.002,0.002,0.003,0.003,0.003,0.006,0.006,0.006])

    col1=fits.Column(name='CHANNEL',format='I',array=chan)
    col2=fits.Column(name='BAND',format='10A',array=bnd)
    col3=fits.Column(name='WAVEMIN',format='E',array=wmin, unit='micron')
    col4=fits.Column(name='WAVEMAX',format='E',array=wmax, unit='micron')
    col5=fits.Column(name='SPAXELSIZE',format='E',array=spaxsize, unit='arcsec')
    col6=fits.Column(name='SPECTRALSTEP',format='D',array=wsamp, unit='micron')

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6])
    hdu.header['EXTNAME']='CUBEPAR'
    
    return hdu

#############################

# Create 2nd extension (CUBEPAR_MSM) with per-band Modified Shepard (1/r2 weight) parameters

def make_ext2():
    chan=np.array([1,1,1,2,2,2,3,3,3,4,4,4])
    bnd=np.array(['SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG'])
    roispat=np.array([0.10,0.10,0.10,0.15,0.15,0.15,0.20,0.20,0.20,0.40,0.40,0.40])
    roispec=np.array([0.001,0.001,0.001,0.002,0.002,0.002,0.003,0.003,0.003,0.006,0.006,0.006])
    power=np.array([2,2,2,2,2,2,2,2,2,2,2,2])
    softrad=np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])

    col1=fits.Column(name='CHANNEL',format='I',array=chan)
    col2=fits.Column(name='BAND',format='10A',array=bnd)
    col3=fits.Column(name='ROISPATIAL',format='E',array=roispat, unit='arcsec')
    col4=fits.Column(name='ROISPECTRAL',format='E',array=roispec, unit='micron')
    col5=fits.Column(name='POWER',format='I',array=power)
    col6=fits.Column(name='SOFTRAD',format='E',array=softrad, unit='arcsec')

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6])
    hdu.header['EXTNAME']='CUBEPAR_MSM'
    
    return hdu

#############################

# Create 3rd extension (MULTICHANNEL_MSM) with multichannel Modified Shepard (1/r weight) parameters

def make_ext3():
    # Define the multiextension wavelength solution
    finalwave=mrs_multiwave()
    nelem=len(finalwave)

    # Linear relation of spatial roi with wavelength
    roispat=((0.47762609-0.076412330)/(28.4186-4.89000))*(finalwave-4.890)+0.076412330
    # Linear relation of spectral roi with wavelength
    roispec=((0.0082460508 - 0.00067999235)/(28.4186-4.89000))*(finalwave-4.890)+0.00067999235
    # Power is 2 at all wavelengths
    power=np.ones(nelem,dtype=int)*2
    # Softening radius is 0.01 at all wavelengths
    softrad=np.ones(nelem)*0.01

    col1=fits.Column(name='WAVELENGTH',format='D',array=finalwave, unit='micron')
    col2=fits.Column(name='ROISPATIAL',format='E',array=roispat, unit='arcsec')
    col3=fits.Column(name='ROISPECTRAL',format='E',array=roispec, unit='micron')
    col4=fits.Column(name='POWER',format='I',array=power)
    col5=fits.Column(name='SOFTRAD',format='E',array=softrad, unit='arcsec')

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5])
    hdu.header['EXTNAME']='MULTICHANNEL_MSM'
    
    return hdu

#############################

# Generate the non-linear MRS multiwavelength lambda vector

def mrs_multiwave():
    # Model the spectral resolution as a function of wavelength
    # with a linear function
    testlam=np.arange(10000)/10000.*23.+5.
    # Empirical linear fit
    testres=-142.53*testlam+4490.75

    # Target wavelength sampling is twice per resolution element
    testsamp=testlam/(2*testres)

    # Set up the generating equation for our wavelength grid
    nwave=10000 # Start with 10,000 samples and shrink later as necessary
    lam=np.zeros(nwave) # Temporary wavelength vector
    lam[0]=4.890# Starting wavelength
    dlam=np.zeros(nwave) # Temporary delta-wavelength vector
    rvec=np.zeros(nwave) # Temporary resolving power

    i=0
    maxwave=28.45
    while (lam[i] <= maxwave):
        rvec[i]=4490.75-142.53*lam[i]
        dlam[i]=lam[i]/(2*rvec[i])
        lam[i+1]=lam[i]+dlam[i]
        i=i+1

    finalwave=lam[0:i]

    return finalwave

#############################

# Create 4th extension (CUBEPAR_EMSM) with per-band Modified Shepard (exponential weight) parameters

def make_ext4():
    chan=np.array([1,1,1,2,2,2,3,3,3,4,4,4])
    bnd=np.array(['SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG'])

    # Rough middle wavelengths
    lamcen=np.array([5.3,6.1,7.1,8.2,9.4,10.8,12.5,14.5,16.75,19.1,22.5,26.0])
    fwhm=lamcen/8.*0.31
    # Kludge ch1 fwhm because non-circular
    fwhm[0],fwhm[1],fwhm[2]=0.31,0.31,0.31

    # Rough guess at exponential scale radius as FWHM/3 (~ 1 sigma)
    scalerad=fwhm/3.
    # Rough guess at ROI limiting region
    roispat=fwhm*1.3
    # Rough guess at ROI spectral region
    roispec=np.array([0.0025,0.0025,0.0025,0.005,0.005,0.005,0.007,0.007,0.007,0.014,0.014,0.014])

    col1=fits.Column(name='CHANNEL',format='I',array=chan)
    col2=fits.Column(name='BAND',format='10A',array=bnd)
    col3=fits.Column(name='ROISPATIAL',format='E',array=roispat, unit='arcsec')
    col4=fits.Column(name='ROISPECTRAL',format='E',array=roispec, unit='micron')
    col5=fits.Column(name='SCALERAD',format='E',array=scalerad, unit='arcsec')

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5])
    hdu.header['EXTNAME']='CUBEPAR_EMSM'
    
    return hdu

#############################

# Create 5th extension (MULTICHANNEL_EMSM) with multichannel Modified Shepard (exponential weight) parameters

def make_ext5():
    # Define the multiextension wavelength solution
    finalwave=mrs_multiwave()
    nelem=len(finalwave)

    # Rough FWHM
    fwhm=finalwave/8.*0.31
    # Kludge fwhm below 8 microns b/c non-circular
    indx=np.where(finalwave < 8.0)
    fwhm[indx]=0.31

    # Rough guess at exponential scale radius as FWHM/3 (~ 1 sigma)
    scalerad=fwhm/3.
    # Rough guess at ROI limiting region
    roispat=fwhm*1.3
    # Rough guess at ROI spectral region
    roispec=((0.014 - 0.0025)/(28.4186-4.89000))*(finalwave-4.890)+0.0025

    col1=fits.Column(name='WAVELENGTH',format='D',array=finalwave, unit='micron')
    col2=fits.Column(name='ROISPATIAL',format='E',array=roispat, unit='arcsec')
    col3=fits.Column(name='ROISPECTRAL',format='E',array=roispec, unit='micron')
    col4=fits.Column(name='SCALERAD',format='E',array=scalerad, unit='arcsec')

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4])
    hdu.header['EXTNAME']='MULTICHANNEL_EMSM'
    
    return hdu
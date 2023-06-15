#
"""
Python tools for creating the MIRI MRS cubepar parameter files.
These define the values of things like spaxel size, wavelength solution,
and weight function parameters for the JWST pipeline cube building algorithm.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
Mid-2018  IDL version written by Beth Sargent (sargent@stsci.edu)
18-Jul-2019  Convert to python (D. Law)
19-Jul-2019  Add exponential weight extensions (D. Law)
12-Dec-2019  Modify weights based on pipeline testing (D. Law)
08-Jan-2021  Tweak cube wavelength ranges to account for isolambda curvature (D. Law)
04-Feb-2021  Add cross-dichroic information (D. Law)
05-Aug-2021  Add DRIZ mode multiband information (D. Law)
07-Jun-2022  Fix cross-grating assignments (D. Law)
21-Oct-2022  Adjust to flight spectral resolution and 4C cutoff (D. Law)

"""

from astropy.io import fits
from astropy.time import Time
import datetime
import os as os
import numpy as np
import miricoord.mrs.mrs_tools as mt
import pdb

#############################

# This routine is the master function to make the reference file
def make_cubepar():
    # miricoord version
    print('miricoord version: '+mt.version())
    
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
    print('Making 0th extension')
    hdu0=make_ext0(now,filename)

    # Create first extension (CUBEPAR: basic spaxel size, etc)
    print('Making 1st extension')
    hdu1=make_ext1()

    # Create second extension (CUBEPAR_MSM: per-band 1/r2 weights and roi)
    print('Making 2nd extension')
    hdu2=make_ext2()

    # Create third extension (MULTICHANNEL_MSM: multichannel 1/r2 weights and roi)
    print('Making 3rd extension')
    hdu3=make_ext3()

    # Create the fourth extension (CUBEPAR_EMSM: per-band exponential weights and roi)
    print('Making 4th extension')
    hdu4=make_ext4()
    
    # Create the fifth extension (MULTICHANNEL_EMSM: multichannel exponential weights and roi)
    print('Making 5th extension')
    hdu5=make_ext5()

    # Create the sixth extension (MULTICHANNEL_DRIZ: multichannel wavelengths)
    print('Making 6th extension')
    hdu6=make_ext6()
    
    hdul=fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4,hdu5,hdu6])
    hdul.writeto(outfile,overwrite=True)
    print('Wrote file '+outfile)

#############################

# Compute the min/max wavelength to use for cube building for a given channel

def waveminmax(channel,**kwargs):
    # To save time don't recompute the wavelength image if passed in one
    if 'waveimage' in kwargs:
        wimg=kwargs['waveimage']
    else:
        wimg=mt.waveimage(channel)
        
    distfile=fits.open(mt.get_fitsreffile(channel))
    slicemap=distfile['Slice_Number'].data
    # Unless otherwise specified, use the most permissive throughput slice map
    if 'mapplane' in kwargs:
        slicemap=slicemap[kwargs['mapplane'],:,:]
    else:
        slicemap=slicemap[0,:,:]

    # Zero out where wavelengths are zero
    indx=np.where(wimg < 1.)
    slicemap[indx]=0

    # Find unique slice numbers
    slicenum=np.unique(slicemap)
    # Cut out the zero value
    slicenum=slicenum[1:]
    nslice=len(slicenum)

    thislmin=np.zeros(nslice)
    thislmax=np.zeros(nslice)

    # Ignore the top and bottom 3 rows of the detector as these will be masked with
    # DO_NOT_USE in actual data
    # Look at 4th-to-top and 4th-to-bottom rows of the detector
    wimg_row1 = wimg[3,:]
    wimg_row2 = wimg[-4,:]
    slice_row1 = slicemap[3,:]
    slice_row2 = slicemap[-4,:]

    # Detectors have different orientations.  Ensure that row1 is shorter wavelength than row2
    if (np.max(wimg_row1) > np.max(wimg_row2)):
        wimg_row1, wimg_row2 = wimg_row2, wimg_row1
        slice_row1, slice_row2 = slice_row2, slice_row1

    # Max and min wavelength in each slice accounting for the curvature of isolambda
    for jj in range(0,nslice):
        indx=np.where(slice_row1 == slicenum[jj])
        thislmin[jj]=np.max(wimg_row1[indx])
        indx = np.where(slice_row2 == slicenum[jj])
        thislmax[jj]=np.min(wimg_row2[indx])

    # Ensure overall min/max wavelengths are covered for all slices
    lmin=np.max(thislmin)
    lmax=np.min(thislmax)

    return lmin,lmax

#############################

# Compute the wavelength extrema for a given channel
# (useful to set an inclusive simulation range)

def waveextrema(channel):
    wimg=mt.waveimage(channel)
    distfile=fits.open(mt.get_fitsreffile(channel))
    slicemap=distfile['Slice_Number'].data
    slicemap=slicemap[0,:,:]# Most permissive slice mask

    # Zero out where wavelengths are zero
    indx=np.where(wimg < 1.)
    slicemap[indx]=0

    # Find unique slice numbers
    slicenum=np.unique(slicemap)
    # Cut out the zero value
    slicenum=slicenum[1:]
    nslice=len(slicenum)

    thislmin=np.zeros(nslice)
    thislmax=np.zeros(nslice)
    # Max and min wavelength in each slice
    for jj in range(0,nslice):
        indx=np.where(slicemap == slicenum[jj])
        thislmin[jj]=np.min(wimg[indx])
        thislmax[jj]=np.max(wimg[indx])

    # Extreme values across all slices
    lmax=np.max(thislmax)
    lmin=np.min(thislmin)
    
    return lmin,lmax

#############################

# Create blank primary extension and header with base instrument information

def make_ext0(now,thisfile):
    hdu=fits.PrimaryHDU()
    
    hdu.header['DATE']=now.value

    hdu.header['REFTYPE']='CUBEPAR'
    hdu.header['DESCRIP']='Default IFU Cube Sampling and weight parameters'
    hdu.header['PEDIGREE']='INFLIGHT 2022-06-08 2022-06-08'
    hdu.header['DATAMODL']='MiriIFUCubeParsModel'
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
    hdu.header['HISTORY']='IFU Cube defaults'
    hdu.header['HISTORY']='DOCUMENT: TBD'
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/cubepar/make_cubepar.py'
    hdu.header['HISTORY']='DATA USED: Simulated data created by D. Law'
    hdu.header['HISTORY']='DIFFERENCES: Migrated to python creation code.'
    hdu.header['HISTORY']='DIFFERENCES: Add support for e^(-r) weight function.'
    hdu.header['HISTORY']='Set parameters for 12 bands based on simulations.'
    hdu.header['HISTORY']='Tweak wavelength ranges to account for isolambda tilt within slices.'
    hdu.header['HISTORY']='Add cross-dichroic configurations.'
    hdu.header['HISTORY']='Add support for multiband drizzling.'
    hdu.header['HISTORY']='June 28 2022: Update for MRS FLT-2'
    hdu.header['HISTORY']='Aug 27 2022: Update for MRS FLT-4'
    hdu.header['HISTORY']='Nov 3 2022: Update 4C cutoff, wavelength sampling, and multiband wavelength solution'
    return hdu

#############################

# Create 1st extension (CUBEPAR) with basic cube setup information

def make_ext1():    
    chan=np.array([1,1,1,2,2,2,3,3,3,4,4,4])
    bnd=np.array(['SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG','SHORT','MEDIUM','LONG'])

    wmin1A,wmax1A=waveminmax('1A')
    wmin1B,wmax1B=waveminmax('1B')
    wmin1C,wmax1C=waveminmax('1C')
    wmin2A,wmax2A=waveminmax('2A')
    wmin2B,wmax2B=waveminmax('2B')
    wmin2C,wmax2C=waveminmax('2C')
    wmin3A,wmax3A=waveminmax('3A')
    wmin3B,wmax3B=waveminmax('3B')
    wmin3C,wmax3C=waveminmax('3C')
    wmin4A,wmax4A=waveminmax('4A')
    wmin4B,wmax4B=waveminmax('4B')
    wmin4C,wmax4C=waveminmax('4C')

    wmin=np.array([wmin1A,wmin1B,wmin1C,wmin2A,wmin2B,wmin2C,wmin3A,wmin3B,wmin3C,wmin4A,wmin4B,wmin4C])
    wmax=np.array([wmax1A,wmax1B,wmax1C,wmax2A,wmax2B,wmax2C,wmax3A,wmax3B,wmax3C,wmax4A,wmax4B,wmax4C])

    # Round wavelength ranges to two decimal places and add a small 10 nm buffer
    # (don't want the VERY edge pixels)
    wmin=np.round(wmin,2)+0.01
    wmax=np.round(wmax,2)-0.01

    # Manual overrides for 4C
    wmin[-1]=24.4
    wmax[-1]=28.7

    spaxsize=np.array([0.13,0.13,0.13,0.17,0.17,0.17,0.20,0.20,0.20,0.35,0.35,0.35])
    wsamp=np.array([0.0008,0.0008,0.0008,0.0013,0.0013,0.0013,0.0025,0.0025,0.0025,0.006,0.006,0.006])

    # Add cross-dichroic information just using the same parameters as each indiv band used
    # Keep in mind that Ch1 and Ch4 are set by the DGAA wheel (first name in band) and
    # Ch2 and Ch3 are set by the DGAB wheel (second name in band)
    xbnd1=np.array(['SHORT-MEDIUM','MEDIUM-LONG','LONG-SHORT','LONG-SHORT','SHORT-MEDIUM','MEDIUM-LONG',
                    'LONG-SHORT','SHORT-MEDIUM','MEDIUM-LONG','SHORT-MEDIUM','MEDIUM-LONG','LONG-SHORT'])
    xbnd2=np.array(['SHORT-LONG','MEDIUM-SHORT','LONG-MEDIUM','MEDIUM-SHORT','LONG-MEDIUM','SHORT-LONG',
                    'MEDIUM-SHORT','LONG-MEDIUM','SHORT-LONG','SHORT-LONG','MEDIUM-SHORT','LONG-MEDIUM'])
    allchan=np.append(chan,np.append(chan,chan))
    allbnd=np.append(bnd,np.append(xbnd1,xbnd2))
    allwmin=np.append(wmin,np.append(wmin,wmin))
    allwmax=np.append(wmax,np.append(wmax,wmax))
    allspaxsize=np.append(spaxsize,np.append(spaxsize,spaxsize))
    allwsamp=np.append(wsamp,np.append(wsamp,wsamp))
    
    col1=fits.Column(name='CHANNEL',format='I',array=allchan)
    col2=fits.Column(name='BAND',format='12A',array=allbnd)
    col3=fits.Column(name='WAVEMIN',format='E',array=allwmin, unit='micron')
    col4=fits.Column(name='WAVEMAX',format='E',array=allwmax, unit='micron')
    col5=fits.Column(name='SPAXELSIZE',format='E',array=allspaxsize, unit='arcsec')
    col6=fits.Column(name='SPECTRALSTEP',format='D',array=allwsamp, unit='micron')
    
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

    # Add cross-dichroic information just using the same parameters as each indiv band used
    # Keep in mind that Ch1 and Ch4 are set by the DGAA wheel (first name in band) and
    # Ch2 and Ch3 are set by the DGAB wheel (second name in band)
    xbnd1=np.array(['SHORT-MEDIUM','MEDIUM-LONG','LONG-SHORT','LONG-SHORT','SHORT-MEDIUM','MEDIUM-LONG',
                    'LONG-SHORT','SHORT-MEDIUM','MEDIUM-LONG','SHORT-MEDIUM','MEDIUM-LONG','LONG-SHORT'])
    xbnd2=np.array(['SHORT-LONG','MEDIUM-SHORT','LONG-MEDIUM','MEDIUM-SHORT','LONG-MEDIUM','SHORT-LONG',
                    'MEDIUM-SHORT','LONG-MEDIUM','SHORT-LONG','SHORT-LONG','MEDIUM-SHORT','LONG-MEDIUM'])

    allchan=np.append(chan,np.append(chan,chan))
    allbnd=np.append(bnd,np.append(xbnd1,xbnd2))
    allroispat=np.append(roispat,np.append(roispat,roispat))
    allroispec=np.append(roispec,np.append(roispec,roispec))
    allpower=np.append(power,np.append(power,power))
    allsoftrad=np.append(softrad,np.append(softrad,softrad))
    
    col1=fits.Column(name='CHANNEL',format='I',array=allchan)
    col2=fits.Column(name='BAND',format='12A',array=allbnd)
    col3=fits.Column(name='ROISPATIAL',format='E',array=allroispat, unit='arcsec')
    col4=fits.Column(name='ROISPECTRAL',format='E',array=allroispec, unit='micron')
    col5=fits.Column(name='POWER',format='I',array=allpower)
    col6=fits.Column(name='SOFTRAD',format='E',array=allsoftrad, unit='arcsec')

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
    # Empirical fit
    tempx=np.array([5,10,12,17,27])
    tempy=np.array([4000,3400,3000,2500,1400])
    test=np.polyfit(tempx,tempy,2)
    model=np.poly1d(test)
    testres=model(testlam)

    # Target wavelength sampling is twice per resolution element
    testsamp=testlam/(2*testres)

    # Set up the generating equation for our wavelength grid
    nwave=12000 # Start with 12,000 samples and shrink later as necessary
    lam=np.zeros(nwave) # Temporary wavelength vector
    lam[0]=4.90# Starting wavelength
    dlam=np.zeros(nwave) # Temporary delta-wavelength vector
    rvec=np.zeros(nwave) # Temporary resolving power

    i=0
    maxwave=27.9
    while (lam[i] <= maxwave):
        rvec[i]=model(lam[i])
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
    roispat=fwhm
    # Rough guess at ROI spectral region
    roispec=np.array([0.0025,0.0025,0.0025,0.005,0.005,0.005,0.007,0.007,0.007,0.014,0.014,0.014])

    # Add cross-dichroic information just using the same parameters as each indiv band used
    # Keep in mind that Ch1 and Ch4 are set by the DGAA wheel (first name in band) and
    # Ch2 and Ch3 are set by the DGAB wheel (second name in band)
    xbnd1=np.array(['SHORT-MEDIUM','MEDIUM-LONG','LONG-SHORT','LONG-SHORT','SHORT-MEDIUM','MEDIUM-LONG',
                    'LONG-SHORT','SHORT-MEDIUM','MEDIUM-LONG','SHORT-MEDIUM','MEDIUM-LONG','LONG-SHORT'])
    xbnd2=np.array(['SHORT-LONG','MEDIUM-SHORT','LONG-MEDIUM','MEDIUM-SHORT','LONG-MEDIUM','SHORT-LONG',
                    'MEDIUM-SHORT','LONG-MEDIUM','SHORT-LONG','SHORT-LONG','MEDIUM-SHORT','LONG-MEDIUM'])

    allchan=np.append(chan,np.append(chan,chan))
    allbnd=np.append(bnd,np.append(xbnd1,xbnd2))
    allroispat=np.append(roispat,np.append(roispat,roispat))
    allroispec=np.append(roispec,np.append(roispec,roispec))
    allscalerad=np.append(scalerad,np.append(scalerad,scalerad))

    col1=fits.Column(name='CHANNEL',format='I',array=allchan)
    col2=fits.Column(name='BAND',format='12A',array=allbnd)
    col3=fits.Column(name='ROISPATIAL',format='E',array=allroispat, unit='arcsec')
    col4=fits.Column(name='ROISPECTRAL',format='E',array=allroispec, unit='micron')
    col5=fits.Column(name='SCALERAD',format='E',array=allscalerad, unit='arcsec')

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
    roispat=fwhm
    # Rough guess at ROI spectral region
    roispec=((0.014 - 0.0025)/(28.4186-4.90000))*(finalwave-4.90)+0.0025

    col1=fits.Column(name='WAVELENGTH',format='D',array=finalwave, unit='micron')
    col2=fits.Column(name='ROISPATIAL',format='E',array=roispat, unit='arcsec')
    col3=fits.Column(name='ROISPECTRAL',format='E',array=roispec, unit='micron')
    col4=fits.Column(name='SCALERAD',format='E',array=scalerad, unit='arcsec')

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4])
    hdu.header['EXTNAME']='MULTICHANNEL_EMSM'
    
    return hdu

#############################

# Create 6th extension (MULTICHANNEL_DRIZ) with multichannel Drizzle parameters
# (just a wavelength solution is needed)

def make_ext6():
    # Define the multiextension wavelength solution
    finalwave=mrs_multiwave()
    nelem=len(finalwave)

    col1=fits.Column(name='WAVELENGTH',format='D',array=finalwave, unit='micron')

    hdu=fits.BinTableHDU.from_columns([col1])
    hdu.header['EXTNAME']='MULTICHANNEL_DRIZ'
    
    return hdu

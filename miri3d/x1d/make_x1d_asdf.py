#
"""
Python tools for creating the MIRI MRS 1d extraction reference files
(extract1d and apcor).
This is the ASDF variant

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
12-Feb-2020  First written (D. Law)
27-Sep-2020  Add in extract1d file as well (D. Law)
29-Oct-2020  Convert to asdf output (D. Law)
"""

import asdf
from astropy.io import fits
from astropy.time import Time
import datetime
import os as os
import numpy as np
import pdb
from matplotlib import pyplot as plt
import miri3d.cubepar.make_cubepar as mc
    
#############################

def make_all():
    make_x1dpar()
    make_apcorrpar()

#############################

def make_x1dpar():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'x1d/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-extract1d-'+str(mjd)+'.asdf'
    outfile=os.path.join(outdir,filename)
    plotname = 'miri-extract1d-'+str(mjd)+'.png'
    outplot = os.path.join(outdir,plotname)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # CDP input directory
    cdp_dir=os.path.expandvars('$MIRIBOX')
    cdp_dir=os.path.join(cdp_dir,'CDP/CDP-7/MRS_APERCORR/')

    # Populate basic metadata
    meta=make_x1d_meta(now,filename)

    # Populate basic operations values
    meta=make_x1d_values(meta)

    # Populate vectors of radii
    meta,data=make_x1d_data(cdp_dir,outplot,meta)

    # Construct file
    tree={
        'meta':meta,
        'data':data
        }
    ff=asdf.AsdfFile(tree)
    
    # Write file
    ff.write_to(outfile,all_array_storage='inline')

#############################

def make_x1d_meta(now,thisfile):
    meta={}
    meta['telescope']='JWST'
    meta['pedigree']='GROUND'
    meta['description']='Default MIRI MRS Extract1d parameters'
    meta['date']=now.value
    meta['reftype']='EXTRACT1D'
    meta['exp_type']='MIR_MRS'
    #meta['modelname']='FM'
    meta['useafter']='2000-01-01T00:00:00'
    meta['version']=int(now.mjd)
    meta['author']='D. Law'
    meta['origin']='STSCI'
    meta['datamodl']='MirMrsExtract1dModel'
    meta['history']='1D Extraction defaults'
    meta['history']+=' DOCUMENT: TBD'
    meta['history']+=' SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py'
    meta['history']+=' DATA USED: CDP-7'
        
    meta['instrument']={
        'name': 'MIRI',
        'channel': 'N/A',
        'band': 'N/A',
        'detector': 'N/A'
        }

    return meta

#############################

def make_x1d_values(meta):
    meta['region_type']='target'
    meta['subtract_background']=True
    meta['method']='subpixel'
    meta['subpixels']=10
    
    return meta

#############################

def make_x1d_data(cdp_dir,outplot,meta):
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
    for file in files:
        fullfile = os.path.join(cdp_dir, file)
        hdu=fits.open(fullfile)
        data=hdu[1].data
        inwave.append(data['wavelength'])
        inap.append(data['a_aperture'])

    # Compile into big vectors
    # Simple polynomial fit to the aperture
    thefit=np.polyfit(np.array(inwave).ravel(),np.array(inap).ravel(),1)
    poly=np.poly1d(thefit)
    radius=poly(waves)

    # Background annulus
    # For now, set inner radius to twice the extraction radius
    inbkg=radius*2
    # And the outer radius to 2.5 times the extraction radius
    outbkg=radius*2.5
    
    plt.plot(inwave,inap,'.')
    plt.plot(waves,radius)
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Extraction Radius (arcsec)')
    plt.savefig(outplot)

    data={
        'wavelength': waves,
        'radius': radius,
        'inner_bkg': inbkg,
        'outer_bkg': outbkg,
        'axis_ratio': axratio,
        'axis_pa': axangle,
        }

    meta['wavelength_units']='micron'
    meta['radius_units']='arcsec'
    meta['inner_bkg_units']='arcsec'
    meta['outer_bkg_units']='arcsec'
    meta['axis_pa_units']='degrees'
        
    return meta,data













#############################

def make_x1dpar_fits():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'x1d/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-extract1d-'+str(mjd)+'.fits'
    outfile=os.path.join(outdir,filename)
    plotname = 'miri-extract1d-'+str(mjd)+'.png'
    outplot = os.path.join(outdir,plotname)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # CDP input directory
    cdp_dir=os.path.expandvars('$MIRIBOX')
    cdp_dir=os.path.join(cdp_dir,'CDP/CDP-7/MRS_APERCORR/')
    
    # Create primary hdu (blank data with header)
    print('Making 0th extension')
    hdu0=make_x1d_ext0(now,filename)

    # Create first extension (basic operation defaults)
    print('Making 1st extension')
    hdu1=make_x1d_ext1(cdp_dir,outplot)

    # Create second extension (vectors of radii)
    print('Making 2nd extension')
    hdu2=make_x1d_ext2(cdp_dir,outplot)
    
    hdul=fits.HDUList([hdu0,hdu1,hdu2])
    hdul.writeto(outfile,overwrite=True)

#############################

def make_apcorrpar_fits():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'x1d/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-apcorrpar-'+str(mjd)+'.fits'
    outfile=os.path.join(outdir,filename)
    plotname = 'miri-apcorrpar-'+str(mjd)+'.png'
    outplot = os.path.join(outdir,plotname)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # CDP input directory
    cdp_dir=os.path.expandvars('$MIRIBOX')
    cdp_dir=os.path.join(cdp_dir,'CDP/CDP-7/MRS_APERCORR/')
    
    # Create primary hdu (blank data with header)
    print('Making 0th extension')
    hdu0=make_apcorr_ext0(now,filename)

    # Create first extension (APCORR)
    print('Making 1st extension')
    hdu1=make_apcorr_ext1(cdp_dir,outplot)

    hdul=fits.HDUList([hdu0,hdu1])
    hdul.writeto(outfile,overwrite=True)
    
#############################

def make_x1d_ext0_fits(now,thisfile):
    hdu=fits.PrimaryHDU()
    
    hdu.header['DATE']=now.value

    hdu.header['REFTYPE']='EXTRACT1D'
    hdu.header['DESCRIP']='Default MIRI MRS Extract1d parameters'
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
    hdu.header['DATAMODL']='MirMrsExtract1dModel'
    hdu.header['HISTORY']='1D Extraction defaults'
    hdu.header['HISTORY']='DOCUMENT: TBD'
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py'
    hdu.header['HISTORY']='DATA USED: CDP-7'
    return hdu
    
#############################

def make_x1d_ext1_fits(now,thisfile):
    val1=np.array(['ANY'])
    val2=np.array(['target'])
    val3=np.array([True])
    val4=np.array(['subpixel'])
    val5=np.array([10])

    col1=fits.Column(name='id',format='10A',array=val1)
    col2=fits.Column(name='region_type',format='10A',array=val2)
    col3=fits.Column(name='subtract_background',format='L',array=val3)
    col4=fits.Column(name='method',format='10A',array=val4)
    col5=fits.Column(name='subpixels',format='I',array=val5)

    hdu=fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5])
    hdu.header['EXTNAME']='PARAMS'
    
    return hdu
    
#############################

def make_x1d_ext2_fits(cdp_dir,outplot):
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
    for file in files:
        fullfile = os.path.join(cdp_dir, file)
        hdu=fits.open(fullfile)
        data=hdu[1].data
        inwave.append(data['wavelength'])
        inap.append(data['a_aperture'])

    # Compile into big vectors
    # Simple polynomial fit to the aperture
    thefit=np.polyfit(np.array(inwave).ravel(),np.array(inap).ravel(),1)
    poly=np.poly1d(thefit)
    radius=poly(waves)

    # Background annulus
    # For now, set inner radius to twice the extraction radius
    inbkg=radius*2
    # And the outer radius to 2.5 times the extraction radius
    outbkg=radius*2.5
    
    plt.plot(inwave,inap,'.')
    plt.plot(waves,radius)
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Extraction Radius (arcsec)')
    plt.savefig(outplot)
    
    col1 = fits.Column(name='WAVELENGTH', format=str(nwave)+'E', unit='micron')
    col2 = fits.Column(name='NELEM_WL', format='I')
    col3 = fits.Column(name='RADIUS', format=str(nwave)+'E', unit='arcsec')
    col4 = fits.Column(name='INNER_BKG', format=str(nwave)+'E', unit='arcsec')
    col5 = fits.Column(name='OUTER_BKG', format=str(nwave)+'E', unit='arcsec')
    col6 = fits.Column(name='AXIS_RATIO', format=str(nwave)+'E')
    col7 = fits.Column(name='AXIS_PA', format=str(nwave)+'E', unit='degrees')
    
    hdu = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7], nrows=1, name="X1D")

    hdu.header['WAVEUNIT'] = ('micron', 'Unit for the WAVELENGTH vector')
    hdu.header['SIZEUNIT'] = ('arcsec', 'Unit for the RADIUS vector')
    
    hdu.data.field("wavelength")[:] = waves
    hdu.data.field("nelem_wl")[:] = nwave
    hdu.data.field("radius")[:] = radius
    hdu.data.field("inner_bkg")[:] = inbkg
    hdu.data.field("outer_bkg")[:] = outbkg
    hdu.data.field("axis_ratio")[:] = axratio
    hdu.data.field("axis_pa")[:] = axangle
    
    return hdu

#############################

def make_apcorr_ext0_fits(now,thisfile):
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
    hdu.header['DATAMODL']='MirMrsApcorrModel'
    hdu.header['HISTORY']='1D Extraction defaults'
    hdu.header['HISTORY']='DOCUMENT: TBD'
    hdu.header['HISTORY']='SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py'
    hdu.header['HISTORY']='DATA USED: CDP-7'
    return hdu
    
#############################

def make_apcorr_ext1_fits(cdp_dir,outplot):
    print('Figuring out wavelength ranges')
    wmin1A,_=mc.waveminmax('1A')
    _,wmax4C=mc.waveminmax('4C')

    print('Building tables')

    # Set channel and band info
    chan = np.array(['ANY'])
    band = np.array(['ANY'])

    # Set the output wavelength sampling for the reference file
    waves = np.arange(wmin1A, wmax4C, 0.01)
    nwave = len(waves)

    # We'll set up 3 radius options for now at each wavelength
    # that can later be used to interpolate to different radius choices
    nrad = 3
    # Define placeholder arrays for radius, apcor, and aperr
    radius = np.zeros([nrad, nwave])
    apcor = np.zeros([nrad, nwave])
    aperr = np.zeros([nrad, nwave]) # Placeholder, all zeros currently

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
    thefit = np.polyfit(np.array(inwave).ravel(), np.array(inap).ravel(), 1)
    poly = np.poly1d(thefit)
    # Radius is the evaluation of this polynomial fit at the output wavelengths
    # For now the low/high radius options are notional
    radius[0, :] = poly(waves)/2
    radius[1, :] = poly(waves)
    radius[2, :] = poly(waves)*2

    # At present the CDP aperture-correction factors have unphysical features
    # Therefore do a simple linear fit to the values with wavelength
    thefit=np.polyfit(np.array(inwave).ravel(),np.array(incor).ravel(),1)
    poly=np.poly1d(thefit)
    # Evaluate polynomial fit to get aperture corrections
    # For now all radius options are the same
    apcor[0, :] = poly(waves)/2.# TEMPORARY!!!!
    apcor[1, :] = poly(waves)
    apcor[2, :] = poly(waves)*2.# TEMPORARY!!!!

    plt.plot(inwave,incor,'.')
    plt.plot(waves[:],apcor[1,:])
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Correction factor')
    plt.savefig(outplot)

    # Dimensional output string
    dimstring = '(' + str(nwave) + ',' + str(nrad) + ')'

    col1 = fits.Column(name='CHANNEL', format='10A')
    col2 = fits.Column(name='BAND', format='10A')
    col3 = fits.Column(name='NELEM_RADIUS', format='I')
    col4 = fits.Column(name='NELEM_WL', format='I')
    col5 = fits.Column(name='WAVELENGTH', format=str(nwave)+'E', unit='micron')
    col6 = fits.Column(name='RADIUS', format=str(nrad * nwave)+'E', dim=dimstring, unit='arcsec')
    col7 = fits.Column(name='APCORR', format=str(nrad * nwave)+'E', dim=dimstring)
    col8 = fits.Column(name='APCORR_ERR', format=str(nrad * nwave)+'E', dim=dimstring)

    hdu = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7,col8], nrows=1, name="APCORR")

    hdu.header['WAVEUNIT'] = ('micron', 'Unit for the WAVELENGTH vector')
    hdu.header['SIZEUNIT'] = ('arcsec', 'Unit for the RADIUS vector')

    hdu.data.field("channel")[:] = chan
    hdu.data.field("band")[:] = band
    hdu.data.field("nelem_radius")[:] = nrad
    hdu.data.field("nelem_wl")[:] = nwave
    hdu.data.field("wavelength")[:] = waves
    hdu.data.field("radius")[:] = radius
    hdu.data.field("apcorr")[:] = apcor
    hdu.data.field("apcorr_err")[:] = aperr
    
    return hdu

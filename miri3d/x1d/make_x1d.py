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
from jwst import datamodels
from jwst.datamodels import IFUExtract1dModel
from jwst.datamodels import MirMrsApcorrModel
from jwst.datamodels import util
    
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

    # If there were not already a datamodel we'd have to construct
    # the dictionary by hand.  We will save that routine for future
    # reference, although it can't be read in in quite the same
    # way (datamodels.open works, but results can't be called the same)
    #ff=make_x1d_fromdict(now,cdp_dir,outplot)

    # Read datamodel from file
    model=datamodels.IFUExtract1dModel()
    # Populate basic metadata
    model=make_x1d_meta(model,now)
    # Populate vectors of data
    model=make_x1d_data(model,cdp_dir,outplot)
    # Put the model into an asdf structure
    ff = asdf.AsdfFile(model)

    # Add history info
    ff.add_history_entry('1D Extraction defaults')
    ff.add_history_entry('DOCUMENT: TBD')
    ff.add_history_entry('SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py')
    ff.add_history_entry('DATA USED: CDP-7')
    
    # Write out the file
    ff.write_to(outfile,all_array_storage='inline')
    
#############################

def make_x1d_meta(model,now):
    model.meta.telescope = 'JWST'
    model.meta.pedigree = 'GROUND'
    model.meta.description = 'Default MIRI MRS Extract1d parameters'
    model.meta.date = now.value
    model.meta.reftype = 'EXTRACT1D'
    model.meta.exp_type = 'MIR_MRS'
    model.meta.useafter = '2000-01-01T00:00:00'
    model.meta.version = int(now.mjd)
    model.meta.author = 'D. Law'
    model.meta.origin = 'STSCI'
    model.meta.instrument.name = 'MIRI'

    model.meta.region_type = 'target'
    model.meta.subtract_background = True
    model.meta.method = 'subpixel'
    model.meta.subpixels = 10

    return model

#############################

def make_x1d_data(model,cdp_dir,outplot):
    print('Figuring out wavelength ranges')
    wmin1A,_=mc.waveminmax('1A')
    _,wmax4C=mc.waveminmax('4C')

    print('Building tables')

    # Set up placeholder vectors
    waves=np.arange(wmin1A,wmax4C,0.01,dtype='float32')
    nwave=len(waves)
    radius=np.ones(nwave,dtype='float32')
    inbkg=np.zeros(nwave,dtype='float32')
    outbkg=np.zeros(nwave,dtype='float32')
    axratio=np.ones(nwave,dtype='float32')
    axangle=np.zeros(nwave,dtype='float32')

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

    model.data.wavelength = waves
    model.data.wavelength_units = 'micron'
    model.data.radius = radius
    model.data.radius_units = 'arcsec'
    model.data.inner_bkg = inbkg
    model.data.inner_bkg_units = 'arcsec'
    model.data.outer_bkg = outbkg
    model.data.outer_bkg_units = 'arcsec'
    model.data.axis_ratio = axratio
    model.data.axis_pa = axangle
    model.data.axis_pa_units = 'degrees'
    
    return model

    
#############################

# This routine is just kept around for reference; it's what we'd
# do if we were not using the data model to create the file
def make_x1d_fromdict(now,cdp_dir,outplot):
    meta={}
    meta['telescope']='JWST'
    meta['pedigree']='GROUND'
    meta['description']='Default MIRI MRS Extract1d parameters'
    meta['date']=now.value
    meta['reftype']='EXTRACT1D'
    meta['exp_type']='MIR_MRS'
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
        'name': 'MIRI'
        }
    meta['region_type']='target'
    meta['subtract_background']=True
    meta['method']='subpixel'
    meta['subpixels']=10

    print('Figuring out wavelength ranges')
    wmin1A,_=mc.waveminmax('1A')
    _,wmax4C=mc.waveminmax('4C')

    print('Building tables')

    # Set up placeholder vectors
    waves=np.arange(wmin1A,wmax4C,0.01,dtype='float32')
    nwave=len(waves)
    radius=np.ones(nwave,dtype='float32')
    inbkg=np.zeros(nwave,dtype='float32')
    outbkg=np.zeros(nwave,dtype='float32')
    axratio=np.ones(nwave,dtype='float32')
    axangle=np.zeros(nwave,dtype='float32')

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
        'wavelenth': waves,
        'wavelength_units':'micron',
        'radius': radius,
        'radius_units':'arcsec',
        'inner_bkg': inbkg,
        'inner_bkg_units':'arcsec',
        'outer_bkg': outbkg,
        'outer_bkg_units':'arcsec',
        'axis_ratio': axratio,
        'axis_pa': axangle,
        'axis_pa_units':'degrees',
        }

    tree={
        'meta':meta,
        'data':data
        }
    
    ff=asdf.AsdfFile(tree)
    
    return ff

#############################

def make_apcorrpar():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRI3D_DATA_DIR')
    outdir=os.path.join(data_dir,'x1d/temp/')
    # Set the output filename including an MJD stamp
    now=Time.now()
    now.format='fits'
    mjd=int(now.mjd)
    filename='miri-apcorrpar-'+str(mjd)+'.asdf'
    outfile=os.path.join(outdir,filename)
    plotname = 'miri-apcorrpar-'+str(mjd)+'.png'
    outplot = os.path.join(outdir,plotname)
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # CDP input directory
    cdp_dir=os.path.expandvars('$MIRIBOX')
    cdp_dir=os.path.join(cdp_dir,'CDP/CDP-7/MRS_APERCORR/')

    # Read datamodel from file (not actually the right one, but close enough?)
    model=datamodels.IFUExtract1dModel()
    # Populate basic metadata
    model=make_apcorr_meta(model,now)
    # Populate vectors of data
    model=make_apcorr_data(model,cdp_dir,outplot)
    # Put the model into an asdf structure
    ff = asdf.AsdfFile(model)

    # Add history info
    ff.add_history_entry('1D Extraction defaults')
    ff.add_history_entry('DOCUMENT: TBD')
    ff.add_history_entry('SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py')
    ff.add_history_entry('DATA USED: CDP-7')
    
    # Write out the file
    ff.write_to(outfile,all_array_storage='inline')
    
#############################

def make_apcorr_meta(model,now):
    model.meta.telescope = 'JWST'
    model.meta.pedigree = 'GROUND'
    model.meta.description = 'Default MIRI MRS Aperture correction parameters'
    model.meta.date = now.value
    model.meta.reftype = 'APCORR'
    model.meta.exp_type = 'MIR_MRS'
    model.meta.useafter = '2000-01-01T00:00:00'
    model.meta.version = int(now.mjd)
    model.meta.author = 'D. Law'
    model.meta.origin = 'STSCI'
    model.meta.instrument.name = 'MIRI'
    model.meta.model_type = 'MirMrsApcorrModel'

    return model


#############################

def make_apcorr_data(model,cdp_dir,outplot):
    print('Figuring out wavelength ranges')
    wmin1A,_=mc.waveminmax('1A')
    _,wmax4C=mc.waveminmax('4C')

    print('Building tables')

    # Set channel and band info
    chan = np.array(['ANY'])
    band = np.array(['ANY'])

    # Set the output wavelength sampling for the reference file
    waves = np.arange(wmin1A, wmax4C, 0.01,dtype='float32')
    nwave = len(waves)

    # We'll set up 3 radius options for now at each wavelength
    # that can later be used to interpolate to different radius choices
    nrad = 3
    # Define placeholder arrays for radius, apcor, and aperr
    radius = np.zeros([nrad, nwave],dtype='float32')
    apcor = np.zeros([nrad, nwave],dtype='float32')
    aperr = np.zeros([nrad, nwave],dtype='float32') # Placeholder, all zeros currently

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

    model.data.wavelength = waves
    model.data.wavelength_units = 'micron'
    model.data.radius = radius
    model.data.radius_units = 'arcsec'
    model.data.apcorr = apcor
    model.data.apcorr_err = aperr
    
    return model

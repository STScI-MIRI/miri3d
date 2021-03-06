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
04-Dec-2020  Change schema names (D. Law)
"""

import asdf
from asdf import AsdfFile
from astropy.io import fits
from astropy.time import Time
import datetime
import os as os
import numpy as np
import pdb
import matplotlib as mpl
from matplotlib import pyplot as plt
import miri3d.cubepar.make_cubepar as mc
from jwst import datamodels
from jwst.datamodels import Extract1dIFUModel
from jwst.datamodels import MirMrsApcorrModel
from jwst.datamodels import util
import miricoord.mrs.makesiaf.makesiaf_mrs as mksiaf
    
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

    # Make the reference file dictionary
    ff=make_x1d_fromdict(now,cdp_dir,outplot)

    # Add history info
    ff.add_history_entry('1D Extraction defaults')
    ff.add_history_entry('DOCUMENT: TBD')
    ff.add_history_entry('SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py')
    ff.add_history_entry('DATA USED: CDP-7')
    
    # Write out the file
    ff.write_to(outfile,all_array_storage='inline')
    print('Wrote file ',outfile)

    # now validate this with the schema. If it does not validate an error is returned
    # working on how to return true or something that says "YES IT WORKDED"
    af = asdf.open(outfile, custom_schema="http://stsci.edu/schemas/jwst_datamodel/extract1difu.schema")
    af.validate()
    
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
    meta['exposure']={
        'type': 'MIR_MRS'
        }
    meta['useafter']='2000-01-01T00:00:00'
    meta['version']=int(now.mjd)
    meta['author']='D. Law'
    meta['origin']='STSCI'
    meta['model_type']='Extract1dIFUModel'
    meta['history']='1D Extraction defaults'
    meta['history']+=' DOCUMENT: TBD'
    meta['history']+=' SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py'
    meta['history']+=' DATA USED: CDP-7'
    meta['history']+=' Updated 4/26/21 to decrease background annulus size'
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
    # Note that Ch1 can be much more generous than Ch4; FWHM increases
    # by a factor of 5 from Ch1 to Ch4 but FOV only by a factor of 2.
    # We also should not apply any sudden steps in the annulus size
    # between channels, otherwise that will manifest as a step in the required
    # aperture correction between channels, and we're assuming that it can be
    # smooth with wavelength so everything interpolates from the same table.

    # Therefore, we'll make annuli that shrink linearly (relative to FWHM)
    # with wavelength
    in1,in2 = np.min(radius)*2.5, np.max(radius)*1.02
    out1,out2 = np.min(radius)*3.0, np.max(radius)*1.5
    inbkg=np.float32(np.interp(waves,np.array([np.min(waves),np.max(waves)]),
                                      np.array([in1,in2])))
    outbkg=np.float32(np.interp(waves,np.array([np.min(waves),np.max(waves)]),
                                      np.array([out1,out2])))

    # QA plot that our aperture and annuli look reasonable
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5,5), dpi=150)
    tband=['1A','2A','3A','4A'] # Bands to test
    twave=[7.5,11.75,18,28] # Wavelengths to test
    ax=[ax1,ax2,ax3,ax4]
    for ii in range(0,len(tband)):
        siaf=mksiaf.create_siaf_oneband(tband[ii])
        indx=np.argmin(np.abs(waves-twave[ii]))
        ax[ii].plot(siaf['inscr_v2_corners'],siaf['inscr_v3_corners'],color='#000000',linewidth=2)
        # Circle showing FWHM
        circle = mpl.patches.Circle((siaf['inscr_v2ref'],siaf['inscr_v3ref']), 0.31*twave[ii]/8./2,linewidth=1,edgecolor='black', facecolor=(0, 0, 0, .0125))
        ax[ii].add_artist(circle)
        # Circle showing extraction radius
        circle = mpl.patches.Circle((siaf['inscr_v2ref'],siaf['inscr_v3ref']), radius[indx],linewidth=1,edgecolor='r', facecolor=(0, 0, 0, .0125))
        ax[ii].add_artist(circle)
        # Circles showing annulus
        circle = mpl.patches.Circle((siaf['inscr_v2ref'],siaf['inscr_v3ref']), inbkg[indx],linewidth=1,edgecolor='b', facecolor=(0, 0, 0, .0125))
        ax[ii].add_artist(circle)
        circle = mpl.patches.Circle((siaf['inscr_v2ref'],siaf['inscr_v3ref']), outbkg[indx],linewidth=1,edgecolor='b', facecolor=(0, 0, 0, .0125))
        ax[ii].add_artist(circle)
        ax[ii].set_xlim(-508,-499)
        ax[ii].set_ylim(-324,-315)
        ax[ii].set_xlabel('V2 (arcsec)')
        ax[ii].set_ylabel('V3 (arcsec)')
        ax[ii].set_title(tband[ii])
    plt.tight_layout()
    plt.savefig(str.replace(outplot,'.png','FOV.png'))
    plt.close()

    # QA plot of final values
    plt.plot(inwave,inap,'.')
    plt.plot(waves,radius)
    plt.plot(waves,inbkg,color='red')
    plt.plot(waves,outbkg,color='red')
    plt.grid()
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Extraction Radius (arcsec)')
    plt.savefig(outplot)
    plt.close()

    data={
        'wavelength': waves,
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

    # Make the reference file dictionary
    ff=make_apcorr_fromdict(now,cdp_dir,outplot)
    
    # Add history info
    ff.add_history_entry('1D Extraction defaults')
    ff.add_history_entry('DOCUMENT: TBD')
    ff.add_history_entry('SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py')
    ff.add_history_entry('DATA USED: CDP-7')
    
    # Write out the file
    ff.write_to(outfile,all_array_storage='inline')

    # now validate this with the schema. If it does not validate an error is returned
    # working on how to return true or something that says "YES IT WORKDED"
    af = asdf.open(outfile, custom_schema="http://stsci.edu/schemas/jwst_datamodel/mirmrs_apcorr.schema")
    af.validate()

#############################

def make_apcorr_fromdict(now,cdp_dir,outplot):
    meta={}
    meta['telescope']='JWST'
    meta['pedigree']='GROUND'
    meta['description']='Default MIRI MRS Aperture correction parameters'
    meta['date']=now.value
    meta['reftype']='APCORR'
    meta['exposure']={
        'type': 'MIR_MRS'
        }
    meta['useafter']='2000-01-01T00:00:00'
    meta['version']=int(now.mjd)
    meta['author']='D. Law'
    meta['origin']='STSCI'
    meta['model_type']='MirMrsApcorrModel'
    meta['history']='1D Extraction defaults'
    meta['history']+=' DOCUMENT: TBD'
    meta['history']+=' SOFTWARE: https://github.com/STScI-MIRI/miri3d/tree/master/miri3d/x1d/make_x1d.py'
    meta['history']+=' DATA USED: CDP-7'        
    meta['instrument']={
        'name': 'MIRI'
        }
    
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
    # For now all radius aperture corrections are the same
    apcor[0, :] = poly(waves)
    apcor[1, :] = poly(waves)
    apcor[2, :] = poly(waves)

    plt.plot(inwave,incor,'.')
    plt.plot(waves[:],apcor[1,:])
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Correction factor')
    plt.savefig(outplot)
    plt.close()
    #pdb.set_trace()
    data={
        'wavelength': waves,
        'wavelength_units':'micron',
        'radius': radius,
        'radius_units':'arcsec',
        'apcorr': apcor,
        'apcorr_err': aperr,
        }

    tree={
        'meta':meta,
        'apcorr_table':data
        }
    
    ff=asdf.AsdfFile(tree)
    
    return ff

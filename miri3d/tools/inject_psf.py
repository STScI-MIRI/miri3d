#
"""
Tool for injecting a point source into a MIRI MRS detector image in order to test
cube building algorithm performance.

Uses miricoord for the distortion transforms, and does not depend on the JWST pipeline.
Works by reading in a template Lvl2b file produced with mirisim and run through the JWST
pipeline, and then overwriting SCI extension values and relevant dither header keywords.

Note that output will need to be run through assign_wcs before they can be built
into data cubes using the pipeline (or just use miri3d to do so).

Required input:
detband: Detector configuration to use (e.g., 12A)
dithers: Dither positions to use (undithered, and standard 4-pt dither available)

Example:
inject_psf.main('12A',[1,2,3,4]) would give a standard 4-pt dither in Ch12A.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
15-Nov-2017  IDL version written by David Law (dlaw@stsci.edu)
30-May-2019  Adapted to python (D. Law)
19-Jul-2019  Overhaul to work for more than Ch1A (D. Law)
"""

import os as os
import sys
import math
import numpy as np
from astropy.io import fits
from numpy.testing import assert_allclose
import miricoord.miricoord.mrs.mrs_tools as mt
import miricoord.miricoord.tel.tel_tools as tt
from scipy import ndimage
from astropy.modeling import models, fitting

import pdb

#############################

# Main wrapper script
# This current assumes up to 4 dither positions 
# E.g., pass it [0] to just use an undithered position
# or [1,2,3,4] to use four-point dither.
#
# Input detband is (e.g.) 12A, 34B, etc.

def main(detband,dithers):
    # Set the distortion solution to use
    mt.set_toolversion('cdp8b')

    # Define the bands to use
    if (detband == '12A'):
        left,right='1A','2A'
    elif (detband == '12B'):
        left,right='1B','2B'
    elif (detband == '12C'):
        left,right='1C','2C'
    elif (detband == '34A'):
        left,right='4A','3A'        
    elif (detband == '34B'):
        left,right='4B','3B'
    elif (detband == '34C'):
        left,right='4C','3C'   
    else:
        print('Input detector/band not recognized!')
        
    print('Setting up the dithers')
    
    # Dithers positions 0,1,2,3,4 in Ideal coordinates
    dxidl=np.array([0.,1.13872,-1.02753,1.02942,-1.13622])
    dyidl=np.array([0.,-0.363763,0.294924,-0.291355,0.368474])
    
    # Select desired combination of dither positions
    # Warning, this will fail if we have bad input!
    dxidl=dxidl[dithers]
    dyidl=dyidl[dithers]
    nexp=len(dxidl)

    # MRS reference location is DEFINED for 1A regardless of band in use
    v2ref,v3ref=mt.abtov2v3(0.,0.,'1A')
    
    # Define source coordinates (decl=0 makes life easier)
    raobj=45.0
    decobj=0.0
    # Make life easier by assuming that telescope roll exactly places
    # slices along R.A. for Ch1A (will not be quite as good for other channels)
    # Compute what that roll is
    a2,b2=2.,0. # A location along alpha axis
    v2_2,v3_2=mt.abtov2v3(a2,b2,'1A')
    ra_2,dec_2,_=tt.jwst_v2v3toradec([v2_2],[v3_2],v2ref=v2ref,v3ref=v3ref,raref=raobj,decref=decobj,rollref=0.)
    dra=(ra_2-raobj)*3600.
    ddec=(dec_2-decobj)*3600.
    roll=-(np.arctan2(dra,ddec)*180./np.pi-90.0)

    # Compute the corresponding raref, decref of the dither positions.
    raref=np.zeros(nexp)
    decref=np.zeros(nexp)
    for ii in range(0,nexp):
        temp1,temp2,_=tt.jwst_v2v3toradec([v2ref]-dxidl[ii],[v3ref]+dyidl[ii],v2ref=v2ref,v3ref=v3ref,raref=raobj,decref=decobj,rollref=roll)
        raref[ii]=temp1
        decref[ii]=temp2

    # Values for each exposure
    allexp=np.zeros([nexp,1024,1032])
    
    # Do left half of detector
    print('Working on left half of detector')
    allexp = setvalues(allexp,left,raobj,decobj,roll,dxidl,dyidl)
    
    # Do right half of detector
    print('Working on right half of detector')
    allexp = setvalues(allexp,right,raobj,decobj,roll,dxidl,dyidl)

    # Write the exposures to disk
    print('Writing files')
    basefile=get_template(detband)
    for ii in range(0,nexp):
        thisexp=allexp[ii,:,:]
        newfile='mock'+detband+'-'+str(ii)+'.fits'
        hdu=fits.open(basefile)
        # Hack header WCS
        header=hdu['SCI'].header
        header['V2_REF']=v2ref
        header['V3_REF']=v3ref
        header['RA_REF']=raref[ii]
        header['DEC_REF']=decref[ii]
        header['ROLL_REF']=roll[0]
        hdu['SCI'].header=header
        hdu['SCI'].data=thisexp
        
        hdu.writeto(newfile,overwrite=True)

    print('Done!')
    
#############################

# Read in the appropriate Lvl2b template file for this channel/band.  These are files in which
# much of the actual pixel data has been zeroed out so that the files can be compressed to be
# very small

def get_template(detband):
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if (detband == '12A'):
        file='template_12A.fits.gz'
    elif (detband == '12B'):
        file='template_12B.fits.gz'
    elif (detband == '12C'):
        file='template_12C.fits.gz'
    elif (detband == '34A'):
        file='template_34A.fits.gz'
    elif (detband == '34B'):
        file='template_34B.fits.gz'
    elif (detband == '34C'):
        file='template_34C.fits.gz'
        
    rootdir=os.path.join(rootdir,'data/lvl2btemplate/')
    reffile=os.path.join(rootdir,file)
   
    return reffile

#############################

# Very rough estimate of the PSF FWHM

def rough_fwhm(band):
    if (band == '1A'):
        fwhm=0.31
    elif (band == '1B'):
        fwhm=0.31
    elif (band == '1C'):
        fwhm=0.31
    elif (band == '2A'):
        fwhm=0.31
    elif (band == '2B'):
        fwhm=0.36
    elif (band == '2C'):
        fwhm=0.42
    elif (band == '3A'):
        fwhm=0.48
    elif (band == '3B'):
        fwhm=0.56
    elif (band == '3C'):
        fwhm=0.65
    elif (band == '4A'):
        fwhm=0.75
    elif (band == '4B'):
        fwhm=0.87
    elif (band == '4C'):
        fwhm=1.0
        
    return fwhm
   
#############################

def setvalues(allexp,band,raobj,decobj,roll,dxidl,dyidl):
    # Define the slice width
    swidth=mt.slicewidth(band)
    # PSF fwhm; approximate it by a gaussian for now that is constant in each band
    fwhm_input=rough_fwhm(band)

    print('Setting up the coordinates')
    
    # Define 0-indexed base x and y pixel number (1032x1024 grid)
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    # Convert to base alpha,beta,lambda at pixel center
    values=mt.xytoabl(basex,basey,band)
    basealpha,basebeta=values['alpha'],values['beta']
    baselambda,slicenum=values['lam'],values['slicenum']
    # Convert to base alpha,beta,lambda at pixel lower-left
    valuesl=mt.xytoabl(basex-0.499,basey-0.499,band)
    basealphal,basebetal=valuesl['alpha'],valuesl['beta']
    baselambdal,slicenuml=valuesl['lam'],valuesl['slicenum']
    # Convert to base alpha,beta,lambda at pixel upper-right
    valuesr=mt.xytoabl(basex+0.499,basey+0.499,band)
    basealphar,basebetar=valuesr['alpha'],valuesr['beta']
    baselambdar,slicenumr=valuesr['lam'],valuesr['slicenum']
    
    # Crop to only pixels on a real slice for this channel
    index0=np.where((basealpha > -50)&(basealphal > -50)&(basealphar > -50))
    basex,basey=basex[index0],basey[index0]
    basealpha,basebeta=basealpha[index0],basebeta[index0]
    baselambda,slicenum=baselambda[index0],slicenum[index0]
    basealphal,basebetal=basealphal[index0],basebetal[index0]
    baselambdal,slicenuml=baselambdal[index0],slicenuml[index0]
    basealphar,basebetar=basealphar[index0],basebetar[index0]
    baselambdar,slicenumr=baselambdar[index0],slicenumr[index0]

    # Convert all alpha,beta base locations to v2,v3 base locations
    basev2,basev3=mt.abtov2v3(basealpha,basebeta,band)
    basev2l,basev3l=mt.abtov2v3(basealphal,basebetal,band)
    basev2r,basev3r=mt.abtov2v3(basealphar,basebetar,band)

    print('Setting up the scene')
    
    # Set up the scene
    scene_xsize=10.#arcsec
    scene_ysize=10.#arcsec
    # Scene sampling
    dx,dy=0.001,0.001# arcsec/pixel
    scene_nx=int(scene_xsize/dx)
    scene_ny=int(scene_ysize/dy)
    scene=np.zeros([scene_ny,scene_nx])
    # Put a point source in the middle
    xcen=int(scene_nx/2)
    ycen=int(scene_ny/2)
    scene[xcen,ycen]=1000.0
    # Convolve with gaussian PSF
    scene=ndimage.gaussian_filter(scene,fwhm_input/dx/2.35)
    # Set up header WCS for the scene
    hdu=fits.PrimaryHDU(scene)
    hdu.header['CD1_1']=dx/3600.
    hdu.header['CD2_2']=dy/3600.
    hdu.header['CRPIX1']=xcen+1
    hdu.header['CRPIX2']=ycen+1
    hdu.header['CRVAL1']=raobj
    hdu.header['CRVAL2']=decobj
    hdu.header['CUNIT1']='deg'
    hdu.header['CUNIT2']='deg'
    hdu.header['CTYPE1']='RA---TAN'
    hdu.header['CTYPE2']='DEC--TAN'

    # Write scene to a file for debugging
    hdu.writeto('scene-'+band+'.fits',overwrite=True)

    print('Projecting scene')
    
    # Compute the RA,DEC of pixel 0,0 for later reference
    ra0=raobj-xcen*dx/3600.
    dec0=decobj-ycen*dy/3600.
    
    # MRS reference location
    v2ref,v3ref=mt.abtov2v3(0.,0.,'1A')

    # Set up individual dithered exposures.
    nexp=len(dxidl)
    # Compute the corresponding raref, decref for each exposure
    raref=np.zeros(nexp)
    decref=np.zeros(nexp)
    for ii in range(0,nexp):
        temp1,temp2,_=tt.jwst_v2v3toradec([v2ref]-dxidl[ii],[v3ref]+dyidl[ii],v2ref=v2ref,v3ref=v3ref,raref=raobj,decref=decobj,rollref=roll)
        raref[ii]=temp1
        decref[ii]=temp2

    # Compute values for each exposure
    for ii in range(0,nexp):
        print('Working on exposure',ii)
        thisexp=allexp[ii,:,:]
        print('Doing coordinate projection')
        # Convert central pixel locations to RA/DEC
        ra,dec,_=tt.jwst_v2v3toradec(basev2,basev3,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=roll)
        # Lower-left pixel edges
        ral,decl,_=tt.jwst_v2v3toradec(basev2l,basev3l,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=roll)
        # Upper-right pixel edges
        rar,decr,_=tt.jwst_v2v3toradec(basev2r,basev3r,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=roll)
        # Top/bottom locations
        dec_upper=dec+swidth/2./3600.
        dec_lower=dec-swidth/2./3600.

        npix=len(basev2)
        x1=(np.round((ral-ra0)/(dx/3600.))).astype(int)
        x2=(np.round((rar-ra0)/(dx/3600.))).astype(int)
        y1=(np.round((dec_lower-dec0)/(dy/3600.))).astype(int)
        y2=(np.round((dec_upper-dec0)/(dy/3600.))).astype(int)
        
        print('Doing pixel value computation')
        for jj in range(0,npix):
            x_ll,x_ur=np.min([x1[jj],x2[jj]]),np.max([x1[jj],x2[jj]])+1
            y_ll,y_ur=np.min([y1[jj],y2[jj]]),np.max([y1[jj],y2[jj]])+1
            
            earea=(x_ur-x_ll)*(y_ur-y_ll)*dx*dy# Effective area in arcsec2 for the surface brightness normalization
            thisexp[basey[jj],basex[jj]]=np.sum(scene[y_ll:y_ur,x_ll:x_ur])/earea

    return allexp

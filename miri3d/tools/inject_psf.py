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
psftot: PSF total flux (in units of mJy)
extval: Background surface brightness (units of MJy/sr)

Example:
inject_psf.main('12A',[1,2,3,4],1,1e-7) would give a standard 4-pt dither in Ch12A with a specified PSF total (total=1) and background/extended value (1e-7)
inject_psf.main('12A',[1,2,3,4],0,1) would give a standard 4-pt dither in Ch12A with no specified PSF and an extended source with flat spectrum of 1

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
15-Nov-2017  IDL version written by David Law (dlaw@stsci.edu)
30-May-2019  Adapted to python (D. Law)
19-Jul-2019  Overhaul to work for more than Ch1A (D. Law)
30-Aug-2019  Many tweaks, addition of beta scan pattern (D. Law)
03-Sep-2019  Overhaul approach to create a slice mask on simulated scene (D. Law)
20-Apr-2021  Overhaul approach to create scanning of a FOV and allow single bands (D. Law)
"""

import os as os
import sys
import math
import numpy as np
from astropy.io import fits
from numpy.testing import assert_allclose
import miricoord.mrs.mrs_tools as mt
import miricoord.mrs.makesiaf.makesiaf_mrs as mrssiaf
import miricoord.tel.tel_tools as tt
from scipy import ndimage
from astropy.modeling import models, fitting

import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb

#############################

# Main wrapper script
# This current assumes up to 4 dither positions 
# E.g., pass it [0] to just use an undithered position
# or [1,2,3,4] to use four-point dither.
#
# Input detband is (e.g.) 12A, 34B, etc.

def main(detband,dithers,psftot,extval,scan=False):
    # Set the distortion solution to use
    mt.set_toolversion('cdp8b')
    
    # Define the bands to use
    left, right = 'N/A', 'N/A'

    if ((detband == '1A')or(detband == '12A')):
        left='1A'
    if ((detband == '2A')or(detband == '12A')):
        right='2A'
    if ((detband == '1B')or(detband == '12B')):
        left='1B'
    if ((detband == '2B')or(detband == '12B')):
        right='2B'
    if ((detband == '1C')or(detband == '12C')):
        left='1C'
    if ((detband == '2C')or(detband == '12C')):
        right='2C'

    if ((detband == '3A')or(detband == '34A')):
        right='3A'
    if ((detband == '4A')or(detband == '34A')):
        left='4A'
    if ((detband == '3B')or(detband == '34B')):
        right='3B'
    if ((detband == '4B')or(detband == '34B')):
        left='4B'
    if ((detband == '3C')or(detband == '34C')):
        right='3C'
    if ((detband == '4C')or(detband == '34C')):
        left='4C'

    #########################################################        
        
    print('Setting up the dithers')
    
    # Normal CDP-8b distortion solution dithers from PRDOPSSOC-M-026
    if (scan == False):
        dxidl=np.array([0.,1.094458,-1.012049,0.988069,-1.117844,0.102213,-0.127945,0.008080,-0.034015])
        dyidl=np.array([0.,-0.385616,0.296642,-0.311605,0.371771,-0.485776,0.467512,-0.499923,0.481275])

        # Select desired combination of dither positions
        # Warning, this will fail if we have bad input!
        dxidl=dxidl[dithers]
        dyidl=dyidl[dithers]
        nexp=len(dxidl)
    
    # If the 'scan' option was True, then override the setup to create a scanning grid to sample
    # the field for a given channel.  Note that this also will only populate a single detector at a time!
    if (scan == True):
        if ((detband == '12A')or(detband == '12B')or(detband == '12C')or(detband == '34A')or(detband == '34B')or(detband == '34C')):
            print('Cannot use scan with selected band!')
        # What is the field for this channel?
        chinfo=mrssiaf.create_siaf_oneband(detband)
        minalpha,maxalpha = np.min(chinfo['inscr_alpha_corners']), np.max(chinfo['inscr_alpha_corners'])
        minbeta,maxbeta = np.min(chinfo['inscr_beta_corners']), np.max(chinfo['inscr_beta_corners'])
        # And the slice width
        sw=mt.slicewidth(detband)
        # Number of slices
        nslice=((maxbeta-minbeta)/sw).astype(int)
        # PSF FWHM for spacing
        fwhm=rough_fwhm(detband)
        # First six points are the corners and sides
        alpha1=np.array([minalpha+fwhm,maxalpha-fwhm,minalpha+fwhm,maxalpha-fwhm,minalpha+fwhm,maxalpha-fwhm])
        beta1=np.array([maxbeta-fwhm,maxbeta-fwhm,0,0,minbeta+fwhm,minbeta+fwhm])
        # Next set of points is a scan up alpha=0 for every slice
        alpha2=np.zeros(nslice)
        beta2=np.arange(nslice)*sw+minbeta+sw/2.
        # Concatenate arrays
        alpha=np.concatenate((alpha1,alpha2))
        beta=np.concatenate((beta1,beta2))

        # Convert to v2,v3
        v2,v3 = mt.abtov2v3(alpha,beta,detband)
        v2zero,v3zero = mt.abtov2v3(0,0,detband)
        # Now convert to Ideal coordinate offsets relative to IFU center FOV in this band
        dx,dy=mt.v2v3_to_xyideal(v2,v3)
        dx0,dy0=mt.v2v3_to_xyideal(v2zero,v3zero)
        dxidl=dx-dx0
        dyidl=dy-dy0
        nexp=len(dxidl)

        # Print the points to a file for reference
        pointfile='simpoints'+detband+'.txt'
        print('# alpha beta',file=open(pointfile,"a"))
        for ii in range(0,nexp):
            print(alpha[ii],beta[ii],file=open(pointfile,"a"))
        
        # Plot where the points were for reference
        plotname='qaplot'+detband+'.png'
        plot_qascan(chinfo,detband,v2,v3,filename=plotname)

    #########################################################

    # MRS reference location is DEFINED for 1A regardless of band in use
    v2ref,v3ref=mt.abtov2v3(0.,0.,'1A')
    
    # Define source coordinates (decl=0 makes life easier)
    raobj=45.0
    decobj=0.0
    # Make life easier by assuming that telescope roll exactly places
    # slices along R.A. for Ch1A (will not be quite as good for other channels)
    # Compute what that roll is
    a1,b1=0.,0.
    a2,b2=2.,0. # A location along alpha axis
    v2_1,v3_1=mt.abtov2v3(a1,b1,'1A')
    v2_2,v3_2=mt.abtov2v3(a2,b2,'1A')
    ra_1,dec_1,_=tt.jwst_v2v3toradec([v2_1],[v3_1],v2ref=v2ref,v3ref=v3ref,raref=raobj,decref=decobj,rollref=0.)
    ra_2,dec_2,_=tt.jwst_v2v3toradec([v2_2],[v3_2],v2ref=v2ref,v3ref=v3ref,raref=raobj,decref=decobj,rollref=0.)
    dra=(ra_2-ra_1)*3600.
    ddec=(dec_2-dec_1)*3600.
    roll=-(np.arctan2(dra,ddec)*180./np.pi-90.0)

    # Compute the corresponding raref, decref, rollref of the dither positions.
    raref=np.zeros(nexp)
    decref=np.zeros(nexp)
    rollref=np.zeros(nexp)
    for ii in range(0,nexp):
        temp1,temp2,temp3=tt.jwst_v2v3toradec([v2ref]-dxidl[ii],[v3ref]+dyidl[ii],v2ref=v2ref,v3ref=v3ref,raref=raobj,decref=decobj,rollref=roll)
        raref[ii]=temp1
        decref[ii]=temp2
        rollref[ii]=temp3

    # Values for each exposure
    allexp=np.zeros([nexp,1024,1032])
    allarea=np.zeros([nexp,1024,1032])
    
    # Do left half of detector
    print('Working on left half of detector')
    roi=rough_fwhm(left)*3
    if (left != 'N/A'):
        allexp,allarea = setvalues(allexp,allarea,left,roi,raobj,decobj,raref,decref,rollref,dxidl,dyidl,psftot,extval)
    
    # Do right half of detector
    print('Working on right half of detector')
    roi=rough_fwhm(right)*3
    if (right != 'N/A'):
        allexp,allarea = setvalues(allexp,allarea,right,roi,raobj,decobj,raref,decref,rollref,dxidl,dyidl,psftot,extval)

    # Write the exposures to disk
    print('Writing files')
    basefile=get_template(detband)
    for ii in range(0,nexp):
        thisexp=allexp[ii,:,:]
        thisarea=allarea[ii,:,:]
        newfile='mock'+detband+'-'+str(ii)+'.fits'
        newareafile='mockarea'+detband+'-'+str(ii)+'.fits'
        hdu=fits.open(basefile)
        # Hack header WCS
        primheader=hdu[0].header
        primheader['TARG_RA']=raobj
        primheader['TARG_DEC']=decobj
        header=hdu['SCI'].header
        header['V2_REF']=v2ref
        header['V3_REF']=v3ref
        header['RA_REF']=raref[ii]
        header['DEC_REF']=decref[ii]
        header['ROLL_REF']=rollref[ii]
        hdu['SCI'].header=header
        hdu['SCI'].data=thisexp
        hdu.writeto(newfile,overwrite=True)
        hdu['SCI'].data=thisarea
        # Overwrite any old DQ problems
        hdu['DQ'].data[:]=0
        hdu.writeto(newareafile,overwrite=True)

    print('Done!')

#############################

# QA plot when doing a field scan approach

def plot_qascan(values,channel,v2,v3,**kwargs):
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(8,5),dpi=250)
    
    # Box limits
    if ('xlim') in kwargs:
        xlim=kwargs['xlim']
    else:
        xlim=[values['slice_v2_corners'].max(),values['slice_v2_corners'].min()]
    if ('ylim') in kwargs:
        ylim=kwargs['ylim']
    else:
        ylim=[values['slice_v3_corners'].min(),values['slice_v3_corners'].max()]

    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True)

    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    nslice=len(values['slice_num'])
    for i in range(0,nslice):
        thiscol='C'+str(i%10)
        plt.plot(values['slice_v2_corners'][i],values['slice_v3_corners'][i],c=thiscol)
        #plt.plot(values['slice_v2ref'][i],values['slice_v3ref'][i],marker='x',linestyle='',c=thiscol)
    plt.plot(values['inscr_v2_corners'],values['inscr_v3_corners'],color='#000000',linewidth=2)
    #plt.plot(values['inscr_v2ref'],values['inscr_v3ref'],color='#000000',marker='x',markersize=10,mew=2)
    plt.plot(v2,v3,'d')
    plt.xlabel('v2 (arcsec)')
    plt.ylabel('v3 (arcsec)')
    plt.title(channel)

    # Determine whether we're sending the plot to screen or to a file
    if ('filename' in kwargs):
        plt.savefig(kwargs['filename'])
        plt.close()
    else:
        plt.show()

#############################

# Read in the appropriate Lvl2b template file for this channel/band.  These are files in which
# much of the actual pixel data has been zeroed out so that the files can be compressed to be
# very small

def get_template(detband):
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

    # Try looking for the file in the expected location
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rootdir=os.path.join(rootdir,'data/lvl2btemplate/')
    reffile=os.path.join(rootdir,file)
    if os.path.exists(reffile):
        return reffile
    
    # If that didn't work, look in the system path
    rootdir=sys.prefix
    rootdir=os.path.join(rootdir,'data/lvl2btemplate/')
    reffile=os.path.join(rootdir,file)
    if os.path.exists(reffile):
        return reffile    

    # If that didn't work either, just return what we've got
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

def setvalues(allexp,allarea,band,roi,raobj,decobj,raref,decref,rollref,dxidl,dyidl,psftot,extval):
    # Define the slice width
    swidth=mt.slicewidth(band)
    # PSF fwhm; approximate it by a gaussian for now that is constant in each band
    fwhm_input=rough_fwhm(band)

    print('Setting up the coordinates')
    
    # Define 0-indexed base x and y pixel number (1032x1024 grid)
    if ((band == '1A')or(band == '1B')or(band == '1C')):
        detxmin,detxmax=0,519
    if ((band == '2A')or(band == '2B')or(band == '2C')):
        detxmin,detxmax=508,1032
    if ((band == '3A')or(band == '3B')or(band == '3C')):
        detxmin,detxmax=500,1032
    if ((band == '4A')or(band == '4B')or(band == '4C')):
        detxmin,detxmax=0,500
        
    basex,basey = np.meshgrid(np.arange(detxmin,detxmax),np.arange(1024))
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    
    # Convert to base alpha,beta,lambda at pixel center
    values=mt.xytoabl(basex,basey,band)
    basealpha,basebeta=values['alpha'],values['beta']
    baselambda,slicenum=values['lam'],values['slicenum']

    # Convert to base alpha,beta,lambda at pixel lower-left
    valuesll=mt.xytoabl(basex-0.4999,basey-0.4999,band)
    basealphall,basebetall=valuesll['alpha'],valuesll['beta']-swidth/2.
    baselambdall,slicenumll=valuesll['lam'],valuesll['slicenum']
    # Convert to base alpha,beta,lambda at pixel upper-right
    valuesur=mt.xytoabl(basex+0.4999,basey+0.4999,band)
    basealphaur,basebetaur=valuesur['alpha'],valuesur['beta']+swidth/2.
    baselambdaur,slicenumur=valuesur['lam'],valuesur['slicenum']

    # Convert to base alpha,beta,lambda at pixel upper-left
    valuesul=mt.xytoabl(basex-0.4999,basey+0.4999,band)
    basealphaul,basebetaul=valuesul['alpha'],valuesul['beta']+swidth/2.
    baselambdaul,slicenumul=valuesul['lam'],valuesul['slicenum']
    # Convert to base alpha,beta,lambda at pixel lower-right
    valueslr=mt.xytoabl(basex+0.4999,basey-0.4999,band)
    basealphalr,basebetalr=valueslr['alpha'],valueslr['beta']-swidth/2.
    baselambdalr,slicenumlr=valueslr['lam'],valueslr['slicenum']

    # Crop to only pixels on a real slice for this channel
    temp=basealpha+basealphaul+basealphaur+basealphall+basealphalr
    index0=np.where(temp > -50)
    basex,basey=basex[index0],basey[index0]
    basealpha,basebeta=basealpha[index0],basebeta[index0]
    baselambda,slicenum=baselambda[index0],slicenum[index0]
    basealphall,basealphalr=basealphall[index0],basealphalr[index0]
    basealphaul,basealphaur=basealphaul[index0],basealphaur[index0]
    basebetall,basebetalr=basebetall[index0],basebetalr[index0]
    basebetaul,basebetaur=basebetaul[index0],basebetaur[index0]

    # Convert to v2,v3 projected base locations
    basev2,basev3=mt.abtov2v3(basealpha,basebeta,band)
    basev2ll,basev3ll=mt.abtov2v3(basealphall,basebetall,band)
    basev2ul,basev3ul=mt.abtov2v3(basealphaul,basebetaul,band)
    basev2lr,basev3lr=mt.abtov2v3(basealphalr,basebetalr,band)
    basev2ur,basev3ur=mt.abtov2v3(basealphaur,basebetaur,band)

    print('Setting up the scene')
    
    # Set up the scene
    scene_xsize=10.#arcsec
    scene_ysize=10.#arcsec
    # Scene sampling
    dx,dy=0.002,0.002# arcsec/pixel
    as_per_ster=4.25452e10 # Square arcsec per steradian
    scene_nx=int(scene_xsize/dx)
    scene_ny=int(scene_ysize/dy)
    scene=np.zeros([scene_ny,scene_nx])
    # Scene coordinates in 1d
    scenex,sceney = np.meshgrid(np.arange(scene_ny),np.arange(scene_nx))
    scenex=scenex.reshape(-1)
    sceney=sceney.reshape(-1)
    # Put a point source in the middle
    xcen=int(scene_nx/2)
    ycen=int(scene_ny/2)
    # Insert point source, convert from mJy to MJy
    scene[xcen,ycen]=psftot*1e-9
    # HACK; additional sources
    #scene[xcen+np.fix(0.8/dx).astype(int),ycen+np.fix(0.8/dy).astype(int)]=psftot*1e-9
    #scene[xcen+np.fix(0.8/dx).astype(int),ycen-np.fix(1.2/dy).astype(int)]=psftot*1e-9
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

    # Stick something near-zero but not zero into all light-sensitive pixels
    # just so that we can see where they are
    for jj in range(0,len(basex)):
        allexp[:,basey[jj],basex[jj]]=extval #1e-7
    
    # Compute values for each exposure
    for ii in range(0,nexp):
        print('Working on exposure',ii)
        thisexp=allexp[ii,:,:]
        thisarea=allarea[ii,:,:]
        print('Doing coordinate projection')
        # Convert central pixel locations to RA/DEC
        ra,dec,_=tt.jwst_v2v3toradec(basev2,basev3,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])
        
        # Throw away everything not within the ROI (in arcsec) to save compute time
        dist=np.sqrt((ra-raobj)*(ra-raobj)+(dec-decobj)*(dec-decobj))*3600.
        close=np.where(dist <= roi)
        close=close[0]# silly python thing to get indices
        cbasev2ll,cbasev3ll=basev2ll[close],basev3ll[close]
        cbasev2lr,cbasev3lr=basev2lr[close],basev3lr[close]
        cbasev2ul,cbasev3ul=basev2ul[close],basev3ul[close]
        cbasev2ur,cbasev3ur=basev2ur[close],basev3ur[close]
        cbasex,cbasey=basex[close],basey[close]
        cslicenum=slicenum[close]
        cra=ra[close]
        
        # Lower-left pixel edges
        rall,decll,_=tt.jwst_v2v3toradec(cbasev2ll,cbasev3ll,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])
        # Upper-right pixel edges
        raur,decur,_=tt.jwst_v2v3toradec(cbasev2ur,cbasev3ur,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])
        # Lower-right pixel edges
        ralr,declr,_=tt.jwst_v2v3toradec(cbasev2lr,cbasev3lr,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])
        # Upper-left pixel edges
        raul,decul,_=tt.jwst_v2v3toradec(cbasev2ul,cbasev3ul,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])
        
        npix=len(cra)
        xll=(rall-ra0)/(dx/3600.)
        xlr=(ralr-ra0)/(dx/3600.)
        xul=(raul-ra0)/(dx/3600.)
        xur=(raur-ra0)/(dx/3600.)

        yll=(decll-dec0)/(dy/3600.)
        ylr=(declr-dec0)/(dy/3600.)
        yul=(decul-dec0)/(dy/3600.)
        yur=(decur-dec0)/(dy/3600.)

        # Project slice numbers onto the simulated image and use these instead of beta boundaries 
        # to define when stuff is 'in' a pixel.  Since we're iterating over all lambda we'll ensure there
        # aren't any missing elements, and since we're overwriting each time we make sure there aren't
        # any duplicated elements
        print('Projecting slices')
        smaskimg=scene.copy()
        xleftvec,xrightvec=np.zeros(npix),np.zeros(npix)
        for jj in range(0,npix):
            xtemp1=(xll[jj]+xul[jj])/2.
            xtemp2=(xlr[jj]+xur[jj])/2.
            ytemp1=(yll[jj]+ylr[jj])/2.
            ytemp2=(yul[jj]+yur[jj])/2.
            # Now ytemp3 and ytemp4 are the full-float guaranteed bottom/top respectively
            xtemp3=np.min([xtemp1,xtemp2])
            xtemp4=np.max([xtemp1,xtemp2])
            ytemp3=np.min([ytemp1,ytemp2])
            ytemp4=np.max([ytemp1,ytemp2])
            # Convert to ints
            xleft,xright=np.round(xtemp3).astype(int),np.round(xtemp4).astype(int)
            # Save the x boundaries so we don't have to recalculate later
            xleftvec[jj],xrightvec[jj]=xleft.astype(int),xright.astype(int)
            ybot=np.ceil(ytemp3).astype(int)
            ytop=np.floor(ytemp4).astype(int)+1
            # Write to smaskimg
            smaskimg[ybot:ytop,xleft:xright] = cslicenum[jj]

        xleftvec=xleftvec.astype(int)
        xrightvec=xrightvec.astype(int)
        
        # Create a copy of the scene that is 1 everywhere
        oneimg=scene.copy()
        oneimg[:,:]=1

        print('Doing pixel value computation')
        # Loop over slice numbers
        smaskimg1d=smaskimg.reshape(-1)
        slicevec=np.arange(np.min(slicenum),np.max(slicenum)+1)
        nslice=len(slicevec)
        debugimg=scene.copy()
        debugimg[:,:]=0
        for ii in range(0,nslice):
            # Create a new version of the oneimage where everything not in slice is zeroed out
            thisoneimg=oneimg.copy()
            temp=thisoneimg.reshape(-1)
            badval=np.where(smaskimg1d != slicevec[ii])
            temp[badval]=0 # Cuz python this will propagate back into thisoneimg
            # Create a version of the scene masked by thisoneimg
            thisscene=scene*thisoneimg
            # Find where our detector pixels are in this slice and loop over them
            indx2=np.where(cslicenum == slicevec[ii])
            indx2=indx2[0]
            thisxleft,thisxright=xleftvec[indx2],xrightvec[indx2]
            thisbasex,thisbasey=cbasex[indx2],cbasey[indx2]
            njj=len(indx2)

            # If zero flux in thisscene don't bother with individual calculations
            if (np.max(thisscene > 0)):
                for jj in range(0,njj):
                    area=np.sum(thisoneimg[:,thisxleft[jj]:thisxright[jj]])*dx*dy
                    value=np.sum(thisscene[:,thisxleft[jj]:thisxright[jj]])
                    thisexp[thisbasey[jj],thisbasex[jj]] += value/area*as_per_ster
                    thisarea[thisbasey[jj],thisbasex[jj]]=area
        
    return allexp,allarea

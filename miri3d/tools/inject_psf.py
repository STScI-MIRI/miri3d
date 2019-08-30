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
30-Aug-2019  Many tweaks, addition of beta scan pattern and high-precision option (D. Law)
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

import matplotlib.pyplot as plt
import pdb

#############################

# Main wrapper script
# This current assumes up to 4 dither positions 
# E.g., pass it [0] to just use an undithered position
# or [1,2,3,4] to use four-point dither.
#
# Input detband is (e.g.) 12A, 34B, etc.

def main(detband,dithers,betascan=False,highprec=False):
    # Set the distortion solution to use
    mt.set_toolversion('cdp8b')

    # If doing the high-precision calculation we'll need some extra libraries
    if (highprec == True):
        import shapely
        from shapely.geometry import Polygon, mapping
        import shapely.geometry as sg
        import descartes as dc
    
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

    # If the betascan option was True, use dithers taken from a set scanning every 0.25 slice widths along beta direction
    # Scan set up as dist=np.arange(-3,3.1,0.25), be=dist*slicewidth
    # And then converted to Ideal coordinate offsets
    if (betascan == True):
        dxidl=np.array([-0.07425094, -0.06806336, -0.06187578, -0.0556882 , -0.04950062, -0.04331305, -0.03712547, -0.03093789, -0.02475031, -0.01856273, -0.01237516, -0.00618758, -0.        ,  0.00618758,  0.01237516, 0.01856273,  0.02475031,  0.03093789,  0.03712547,  0.04331305, 0.04950062,  0.0556882 ,  0.06187578,  0.06806336,  0.07425094])
        dyidl=np.array([0.52758708,  0.48362149,  0.4396559 ,  0.39569031,  0.35172472, 0.30775913,  0.26379354,  0.21982795,  0.17586236,  0.13189677, 0.08793118,  0.04396559,  0.        , -0.04396559, -0.08793118, -0.13189677, -0.17586236, -0.21982795, -0.26379354, -0.30775913, -0.35172472, -0.39569031, -0.4396559 , -0.48362149, -0.52758708])
    # Otherwise, use the normal dither pattern
    else:
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
    
    # Do left half of detector
    print('Working on left half of detector')
    roi=rough_fwhm(left)*3
    allexp = setvalues(allexp,left,roi,raobj,decobj,roll,dxidl,dyidl,highprec=highprec)
    
    # Do right half of detector
    print('Working on right half of detector')
    roi=rough_fwhm(right)*3
    allexp = setvalues(allexp,right,roi,raobj,decobj,roll,dxidl,dyidl,highprec=highprec)

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
        header['ROLL_REF']=rollref[ii]
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

def setvalues(allexp,band,roi,raobj,decobj,roll,dxidl,dyidl,highprec=False):
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
    scenetot=1.0
    scene[xcen,ycen]=scenetot
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
    rollref=np.zeros(nexp)
    for ii in range(0,nexp):
        temp1,temp2,temp3=tt.jwst_v2v3toradec([v2ref]-dxidl[ii],[v3ref]+dyidl[ii],v2ref=v2ref,v3ref=v3ref,raref=raobj,decref=decobj,rollref=roll)
        raref[ii]=temp1
        decref[ii]=temp2
        rollref[ii]=temp3
        
    # Compute values for each exposure
    for ii in range(0,nexp):
        print('Working on exposure',ii)
        thisexp=allexp[ii,:,:]
        print('Doing coordinate projection')
        # Convert central pixel locations to RA/DEC
        ra,dec,_=tt.jwst_v2v3toradec(basev2,basev3,v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])

        # Throw away everything not within the ROI (in arcsec) to save time
        dist=np.sqrt((ra-raobj)*(ra-raobj)+(dec-decobj)*(dec-decobj))*3600.
        close=np.where(dist <= roi)
        close=close[0]# silly python thing to get indices
        ra=ra[close]
        dec=dec[close]
        thisbasex=basex[close]
        thisbasey=basey[close]
        
        # Lower-left pixel edges
        rall,decll,_=tt.jwst_v2v3toradec(basev2ll[close],basev3ll[close],v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])
        # Upper-right pixel edges
        raur,decur,_=tt.jwst_v2v3toradec(basev2ur[close],basev3ur[close],v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])
        # Lower-right pixel edges
        ralr,declr,_=tt.jwst_v2v3toradec(basev2lr[close],basev3lr[close],v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])
        # Upper-left pixel edges
        raul,decul,_=tt.jwst_v2v3toradec(basev2ul[close],basev3ul[close],v2ref=v2ref,v3ref=v3ref,raref=raref[ii],decref=decref[ii],rollref=rollref[ii])

        npix=len(ra)
        xll=(rall-ra0)/(dx/3600.)
        xlr=(ralr-ra0)/(dx/3600.)
        xul=(raul-ra0)/(dx/3600.)
        xur=(raur-ra0)/(dx/3600.)

        yll=(decll-dec0)/(dy/3600.)
        ylr=(declr-dec0)/(dy/3600.)
        yul=(decul-dec0)/(dy/3600.)
        yur=(decur-dec0)/(dy/3600.)

        # Strictly this defines a tilted box in the scene, but empirically
        # it's close enough to a simple rectangle (since we aligned the
        # telescope orient properly) that we can simplify it with basic
        # x/y boundaries
        x1=(np.round(xll)).astype(int)
        x2=(np.round(xlr)).astype(int)
        y1=(np.round(yll)).astype(int)
        y2=(np.round(yur)).astype(int)

        #junk=np.where(slicenum == 10)
        #plt.plot(ra[junk],dec_upper[junk],'.',color='b')
        #plt.plot(ra[junk],dec_lower[junk],'.',color='b')
        #plt.plot(ra[junk],decb[junk],'.',color='r')
        #plt.plot(ra[junk],dect[junk],'.',color='g')
        #plt.savefig('test.pdf')
        #pdb.set_trace()   

        # Scene mask for high-precision calculation
        mask=scene.copy()
        
        print('Doing pixel value computation')
        nhigh=0 # Number of high-precision calculations done
        for jj in range(0,npix):         
            xleft,xright=np.min([x1[jj],x2[jj]]),np.max([x1[jj],x2[jj]])+1
            ybot,ytop=np.min([y1[jj],y2[jj]]),np.max([y1[jj],y2[jj]])+1

            # High-precision area
            if (highprec == True):
                polygon = sg.Polygon([(xll[jj], yll[jj]), (xlr[jj],ylr[jj]), (xur[jj], yur[jj]), (xul[jj], yul[jj])])
                earea=polygon.area*dx*dy
            # Basic area
            else:
                earea=(xright-xleft)*(ytop-ybot)*dx*dy# Effective area in arcsec2 for the surface brightness normalization
            
            # Basic summation value from boxcar
            thevalue=np.sum(scene[ybot:ytop,xleft:xright])
            
            # If this pixel contains a large fraction of flux and highprec is selected, do precise calculation
            if ((highprec == True)&(thevalue > 0.02*scenetot)):
                nhigh=nhigh+1
                mask[:,:]=0
                # Temporarily 0.5 near ege of coverage box
                mask[ybot-2:ytop+2,xleft-2:xright+2]=0.5
                # Unity in the middle of coverage box
                mask[ybot+2:ytop-2,xleft+2:xright-2]=1
                # Convert to 1d vector
                mask=mask.reshape(-1)
                totest=(np.where(mask == 0.5))[0]
                ntest=len(totest)

                # Loop over all scene pixels near the edge of the detector pixel projection to compute overlap area
                for kk in range(0,ntest):
                    oi=totest[kk]
                    pixpoly=sg.Polygon([(scenex[oi]-0.5,sceney[oi]-0.5),(scenex[oi]+0.5,sceney[oi]-0.5),(scenex[oi]+0.5,sceney[oi]+0.5),(scenex[oi]-0.5,sceney[oi]+0.5)])
                    overlap=pixpoly.intersection(polygon).area
                    mask[oi]=overlap
                    #if ((overlap > 0)&(overlap < 1)):
                    #    fig = plt.figure()
                    #    ax = fig.add_axes((0.1,0.1,0.8,0.8))
                    #    patch = dc.PolygonPatch(pixpoly, facecolor=[0,0,0.9], edgecolor=[1,1,1], alpha=0.5)
                    #    ax.add_patch(patch)
                    #    patch2 = dc.PolygonPatch(polygon, facecolor=[0,0.9,0.], edgecolor=[1,1,1], alpha=0.5)
                    #    ax.add_patch(patch2)
                    #    plt.xlim([scenex[oi]-10,scenex[oi]+10])
                    #    plt.ylim([sceney[oi]-10,sceney[oi]+10])
                    #    ax.set_aspect(1)
                    #    plt.show()
                    #    pdb.set_trace()
                # Remap mask from 1d array to 2d
                mask=mask.reshape(scene_ny,scene_nx)
                # Multiply it by the scene and integrate
                mscene=mask*scene
                thevalue=np.sum(mscene)
                
            # Normalize by the detector pixel projected area
            thisexp[thisbasey[jj],thisbasex[jj]]=thevalue/earea

        if (highprec == True):
            print('Nhigh precision calcs:',nhigh)
              
    return allexp

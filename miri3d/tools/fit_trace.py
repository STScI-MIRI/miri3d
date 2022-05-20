#
"""
Tool for fitting a trace to the source location in a 2d MRS _cal.fits file.
Loosely based on previous code by Y. Argyriou.

Required input:
file: A 2d MRS _cal.fits file produced by the JWST pipeline.
band: Which band to do the analysis for

Optional input:
recompute: Recompute pixel areas instead of reading from CRDS
nmed: Number of row to median on detector in computations
method: 'meas' returns measured traces, 'model' returns best-fit spline model
verbose: Print log information

Returns: Dictionary of assorted measurements

Example:
fit_trace.fit('det_image_seq1_MIRIFUSHORT_12MEDIUM_cal.fits','1B'))

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
13-Jan-2022  First written
"""

from os.path import exists
import matplotlib.pyplot as plt
import pdb

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats as sigclip
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev
from scipy.interpolate import splrep
from scipy.optimize import curve_fit

import crds.jwst.locate as crds_locate
import miricoord.mrs.mrs_tools as mt
import miricoord.mrs.mrs_pipetools as mpt
import miricoord.tel.tel_tools as tt

#############################

def gauss1d(x, A, mu, sigma, baseline):
    return  A*np.exp(-(x-mu)**2/(2*sigma)**2) + baseline

#############################
#
# Main wrapper script
# Note that this uses independent miricoord distortion code, NOT pipeline code.
# I.e. it might be out of sync!
# Need to specify by hand which side of the detector you want

def fit(file,band,recompute='False',nmed=11,method='meas',verbose=False):
    hdu=fits.open(file)

    # Which channel & band is this data?
    chan_file=hdu[0].header['CHANNEL']
    band_file=hdu[0].header['BAND']
    # Check that band requested actually exists in the file
    band_req,chan_req=mpt.bandchan(band)
    if ((band_req != band_file)or(chan_req != chan_file)):
        print('Error: requested band ',band,' does not exist in file.')
        print('File contains ',chan_file,' ',band_file)
        return 1

    # Compute an array of x,y pixel locations for the entire detector
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))
    
    # Compute the alpha, beta, lambda, and slice number mapping
    if verbose:
        print('Computing distortion mapping')
    temp=mt.xytoabl(basex.ravel(),basey.ravel(),band)
    
    # Reshape to 2d arrays
    alpha=np.reshape(temp['alpha'],basex.shape)
    beta=np.reshape(temp['beta'],basex.shape)
    lam=np.reshape(temp['lam'],basex.shape)
    snum=np.reshape(temp['slicenum'],basex.shape)

    # Define the pixel area array
    arcsec_per_ster=4.255e10
    # Start by looking for the pre-computed array in the photom file
    photom_file=hdu[0].header['R_PHOTOM']
    photom_path=crds_locate.locate_file(photom_file)
    if (exists(photom_path)&(recompute == 'False')):
        if verbose:
            print('Importing pixel areas from ',photom_path)
        photom=fits.open(photom_path)
        pixarea=photom['PIXSIZ'].data # In ster
    # If that can't be found, recompute it (results not identical though)
    else:
        if verbose:
            print("Couldn't find photom file, recomputing pixel areas")
        pixarea=mt.pixarea(band,frame='v2v3')/arcsec_per_ster # Convert to steradians
        
    # Define the science array in units of mJy
    data=hdu['SCI'].data*pixarea*1e9 # Convert from MJy/sr to mJy
    ysize,xsize=data.shape

    # Determine how many slices and their numbers
    slices=np.unique(snum)[1:] # Cut out 0th entry (null slice value)
    nslices=len(slices)

    # Zero out all data not in one of these slices (i.e., other half of detector)
    indx=np.where(snum < 0)
    data[indx]=0.
    
    # Identify the peak slice using a cut along the central row of the detector
    # Median combine down nmed rows to add robustness against noise
    ystart=int(ysize/2 - nmed/2)
    ystop=ystart+nmed
    cut=np.nanmedian(data[ystart:ystop,:],axis=0)
    slicecut=snum[int(ysize/2),:]
    slicesum=np.zeros(nslices)
    # Need to sum the fluxes in each slice to avoid issues where centroid is on pixel boundary vs not
    # in different slices
    for ii in range(0,nslices):
        indx=np.where(slicecut == slices[ii])
        slicesum[ii]=np.nansum(cut[indx])
    peakslice=slices[np.argmax(slicesum)]

    # Get the x trace in the central slice
    xtrace_mid_pass2,xtrace_mid_pass3=trace_slice(peakslice,data,snum,basex,basey,nmed, method,verbose)
    xtrace_lo_pass2,xtrace_lo_pass3=trace_slice(peakslice-1,data,snum,basex,basey,nmed, method,verbose)
    xtrace_hi_pass2,xtrace_hi_pass3=trace_slice(peakslice+1,data,snum,basex,basey,nmed, method,verbose)

    alpha_mid=(mt.xytoabl(xtrace_mid_pass3,basey[:,0],band))['alpha']
    alpha_lo=(mt.xytoabl(xtrace_lo_pass3,basey[:,0],band))['alpha']
    alpha_hi=(mt.xytoabl(xtrace_hi_pass3,basey[:,0],band))['alpha']

    # Final alpha value is the median alpha along the central trace
    good=np.where(alpha_mid > -100) # Ensure we don't use anything that centroided between slices
    alpha=np.nanmedian(alpha_mid[good])

    # Central beta of the slices
    slice_beta=np.zeros(nslices)
    for ii in range(0,nslices):
        # Use any pixel in the slice to get the beta
        indx=np.where(snum == slices[ii])
        slice_beta[ii]=(mt.xytoabl(basex[indx][0],basey[indx][0],band))['beta']

    # We can't simply compare fluxes across slices at a given Y, because
    # there are wavelength offsets and we'd thus see spectral changes
    # not spatial changes in the flux.  Therefore sample at a grid
    # of wavelengths instead

    # Define a wavelength image
    waveim=mt.waveimage(band)
    
    # Define common wavelength range of all slices
    indx=np.where(snum == peakslice)
    allwave=lam[indx]
    minwave,maxwave=np.min(allwave),np.max(allwave)
    dwave=(maxwave-minwave)/ysize*3 # We'll skip every 3 wavelength elements
    # Cut out ends
    minwave,maxwave=minwave+dwave*100,maxwave-dwave*100
    wavevec=np.arange(minwave,maxwave,dwave)
    nwave=len(wavevec)

    ftemp=np.zeros(nslices)
    bcen_vec=np.zeros(nwave)
    bwid_vec=np.zeros(nwave)
    for ii in range(0,nwave):
        ftemp[:]=0.
        for jj in range(0,nslices):
            # We can't trivially fit the trace in each slice either, because it's too faint to
            # see in most, and the width changes from slice to slice.
            # Therefore just sum in a wavelength box.
            indx=np.where((waveim > wavevec[ii]-dwave/2.)&(waveim <= wavevec[ii]+dwave/2.)&(snum == slices[jj]))
            ftemp[jj]=np.nansum(data[indx])
        # Initial guess at fit parameters
        p0=[ftemp.max(),0.,0.5,0.]
        # Bounds for fit parameters
        bound_low=[0.,-10,0,-ftemp.max()]
        bound_hi=[10*np.max(ftemp),10,10,ftemp.max()]
        # Do the fit
        try:
            popt,_ = curve_fit(gauss1d, slice_beta, ftemp, p0=p0, bounds=(bound_low,bound_hi), method='trf')
        except:
            popt=p0
        bcen_vec[ii]=popt[1]
        bwid_vec[ii]=popt[2]

    # Final beta is the median of the values measured at various wavelengths
    beta=np.nanmedian(bcen_vec)

    # Convert final alpha,beta coordinates to v2,v3
    v2,v3=mt.abtov2v3(alpha,beta,band)
    # And convert to RA, DEC
    ra,dec,_=tt.jwst_v2v3toradec([v2],[v3],hdr=hdu[1].header)

    # Define a dictionary to return
    values=dict();
    values['alpha']=alpha
    values['beta']=beta
    values['v2']=v2
    values['v3']=v3
    values['ra']=ra
    values['dec']=dec
    values['beta_vec']=bcen_vec
    values['alpha_vec']=alpha_mid
    values['xtrace2']=xtrace_mid_pass2
    values['xtrace3']=xtrace_mid_pass3

    return values

##########

def trace_slice(thisslice,data,snum,basex,basey,nmed,method,verbose):
    # Zero out everything outside the peak slice
    indx=np.where(snum == thisslice)
    data_slice=data*0.
    data_slice[indx]=data[indx]
    ysize,xsize=data.shape
    xmin,xmax=np.min(basex[indx]),np.max(basex[indx])

    ###################
    
    # First pass for x locations in this slice;
    if verbose:
        print('First pass trace fitting')
    xcen_pass1=np.zeros(ysize)
    for ii in range(0,ysize):
        ystart = max(0,int(ii-nmed/2))
        ystop = min(ysize,ystart+nmed)
        cut=np.nanmedian(data_slice[ystart:ystop,:],axis=0)
        xcen_pass1[ii]=np.argmax(cut)
    # Clean up any bad values by looking for 3sigma outliers
    # and replacing them with the median value
    rms,med=np.nanstd(xcen_pass1),np.nanmedian(xcen_pass1)
    indx=np.where((xcen_pass1 < med-3*rms) | (xcen_pass1 > med+3*rms))
    xcen_pass1[indx]=med
    xwid_pass1=np.ones(ysize) # First pass width is 1 pixel

    ###################
    
    # Second pass for x locations along the trace within this slice
    if verbose:
        print('Second pass trace fitting')
    xcen_pass2=np.zeros(ysize)
    xwid_pass2=np.zeros(ysize)
    for ii in range(0,ysize):
        xtemp=np.arange(xmin,xmax,1)
        ftemp=data_slice[ii,xtemp]
        
        # Initial guess at fit parameters
        p0=[ftemp.max(),xcen_pass1[ii],xwid_pass1[ii],0.]
        # Bounds for fit parameters
        bound_low=[0.,xcen_pass1[ii]-3*rms,0,-ftemp.max()]
        bound_hi=[10*np.max(ftemp),xcen_pass1[ii]+3*rms,10,ftemp.max()]
        # Do the fit
        try:
            popt,_ = curve_fit(gauss1d, xtemp, ftemp, p0=p0, bounds=(bound_low,bound_hi), method='trf')
        except:
            popt=p0
        xcen_pass2[ii]=popt[1]
        xwid_pass2[ii]=popt[2]

    ###################
        
    # Third pass for x location; use a fixed profile width
    twidth=np.nanmedian(xwid_pass2)
    if verbose:
        print('Third pass trace fitting, median trace width ',twidth, ' pixels')
    xcen_pass3=np.zeros(ysize)
    for ii in range(0,ysize):
        xtemp=np.arange(xmin,xmax,1)
        ftemp=data_slice[ii,xtemp]
        
        # Initial guess at fit parameters
        p0=[ftemp.max(),xcen_pass2[ii],twidth,0.]
        # Bounds for fit parameters
        bound_low=[0.,xcen_pass2[ii]-3*rms,twidth*0.999,-ftemp.max()]
        bound_hi=[10*np.max(ftemp),xcen_pass2[ii]+3*rms,twidth*1.001,ftemp.max()]
        # Do the fit
        try:
            popt,_ = curve_fit(gauss1d, xtemp, ftemp, p0=p0, bounds=(bound_low,bound_hi), method='trf')
        except:
            popt=p0
        xcen_pass3[ii]=popt[1]

    # Clean up the fit to remove outliers
    qual=np.ones(ysize)
    # Low order polynomial fit to find the worst outliers using plain RMS
    fit=np.polyfit(basey[:,0],xcen_pass3,2)
    temp=np.polyval(fit,basey[:,0])
    indx=np.where(np.abs(xcen_pass3-temp) > 3*np.nanstd(xcen_pass3-temp))
    qual[indx]=0
    good=(np.where(qual == 1))[0]
    # Another fit to find lesser outliers using sigma-clipped RMS
    fit=np.polyfit(basey[good,0],xcen_pass3[good],2)
    temp=np.polyval(fit,basey[:,0])
    indx=np.where(np.abs(xcen_pass3-temp) > 3*(sigclip(xcen_pass3-temp)[2]))
    qual[indx]=0
    # Find any nan values and set them to bad quality
    indx=np.where(np.isfinite(xcen_pass3) != True)
    qual[indx]=0

    
    # Look for bad failures (e.g., steps that occur if the source is at the edge
    # of the field)
    diff=xcen_pass3-temp
    good=(np.where(qual == 1))[0]
    rms=np.nanstd(diff[good])

    # Replace bad points with the simple polynomial fit
    bad=np.where(qual == 0)
    xcen_pass3[bad]=temp[bad]

    # Update to a spline fit if the data is good enough
    
    # If rms of the GOOD point fit is over 0.3 pixels don't do spline fitting
    if (rms > 0.3):
        print('WARNING: Alpha trace is poor!  Source at edge of field?')
    else:
        # Spline fit
        spl=UnivariateSpline(basey[:,0],xcen_pass3,w=None,s=10)# Not sure about this smoothing factor
        model=spl(basey[:,0])
        # Replace bad values
        bad=np.where(qual == 0)
        xcen_pass3[bad]=model[bad]
        # If method='model' then return the model itself for everything
        if (method == 'model'):
            xcen_pass3=model
            
    return xcen_pass2,xcen_pass3

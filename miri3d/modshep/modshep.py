#
"""
Standalone python implementation of the modified Shepard cube-building algorithm
for the MIRI MRS.  Uses miricoord for the distortion transforms, and does not
depend on the JWST pipeline.

wtype is the type of weighting to use.
wtype=1 is 1/r
wtype=2 is 1/r**2
wtype=3 is e^(-r)

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
01-Apr-2016  IDL version written by David Law (dlaw@stsci.edu)
08-Apr-2019  Adapted to python by Yannis Argyriou
22-May-2019  Adapted from notebook format (D. Law)
25-Jul-2019  Add rough guesses for bands 1B-4C (D. Law)
"""

import os as os
import sys
import math
import numpy as np
from astropy.io import fits
from numpy.testing import assert_allclose
import miricoord.miricoord.mrs.mrs_tools as mt
import miricoord.miricoord.tel.tel_tools as tt

import pdb

#############################

def setcube(filenames,band,wtype=1,**kwargs):
    # Set the distortion solution to use
    mt.set_toolversion('cdp8b')

    nfiles=len(filenames)
    indir = os.path.dirname(os.path.realpath(filenames[0]))
    outdir=indir+'/pystack/'
    if os.path.exists(outdir) is False:
        os.system('mkdir {}'.format(outdir))
    outcube=outdir+'pycube'+band+'.fits'
    outslice=outdir+'slice.fits'
    outcollapse=outdir+'collapse.fits'
    
    # Basic header information from the first file
    head0=fits.open(filenames[0])[0].header
    head1=fits.open(filenames[0])[1].header

    # simulated or ground-test data?
    if (head0['ORIGIN'] == 'MIRI European Consortium'):
        obs_type = 'mirisim'
        databand=head0['BAND']
        datadet=head0['DETECTOR'] # MIRIFUSHORT or MIRIFULONG
    else:
        print('Non-mirisim not implemented!')
        sys.exit(-1)

    # Band-specific cube-building parameters
    if (band == '1A'):
          xmin=8# Minimum x pixel
          xmax=509# Maximum x pixel
          lammin=4.89# Minimum wavelength
          lammax=5.74# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.1
          rlim_arcsec=0.4
          rlimz_mic=0.0025
          ps_x=0.13# arcsec
          ps_y=0.13# arcsec
          ps_z=0.001# micron
    elif (band == '1B'):
          xmin=8# Minimum x pixel
          xmax=509# Maximum x pixel
          lammin=5.64# Minimum wavelength
          lammax=6.62# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.1
          rlim_arcsec=0.4
          rlimz_mic=0.0025
          ps_x=0.13# arcsec
          ps_y=0.13# arcsec
          ps_z=0.001# micron
    elif (band == '1C'):
          xmin=8# Minimum x pixel
          xmax=509# Maximum x pixel
          lammin=6.42# Minimum wavelength
          lammax=7.51# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.1
          rlim_arcsec=0.4
          rlimz_mic=0.0025
          ps_x=0.13# arcsec
          ps_y=0.13# arcsec
          ps_z=0.001# micron
    elif (band == '2A'):
          xmin=510# Minimum x pixel
          xmax=1025# Maximum x pixel
          lammin=7.49# Minimum wavelength
          lammax=8.75# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.15
          rlim_arcsec=0.6
          rlimz_mic=0.005
          ps_x=0.17# arcsec
          ps_y=0.17# arcsec
          ps_z=0.002# micron
    elif (band == '2B'):
          xmin=510# Minimum x pixel
          xmax=1025# Maximum x pixel
          lammin=8.72# Minimum wavelength
          lammax=10.22# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.15
          rlim_arcsec=0.6
          rlimz_mic=0.005
          ps_x=0.17# arcsec
          ps_y=0.17# arcsec
          ps_z=0.002# micron
    elif (band == '2C'):
          xmin=510# Minimum x pixel
          xmax=1025# Maximum x pixel
          lammin=10.03# Minimum wavelength
          lammax=11.74# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.15
          rlim_arcsec=0.6
          rlimz_mic=0.005
          ps_x=0.17# arcsec
          ps_y=0.17# arcsec
          ps_z=0.002# micron
    elif (band == '3A'):
          xmin=510# Minimum x pixel
          xmax=1025# Maximum x pixel
          lammin=11.53# Minimum wavelength
          lammax=13.48# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.2
          rlim_arcsec=0.9
          rlimz_mic=0.007
          ps_x=0.2# arcsec
          ps_y=0.2# arcsec
          ps_z=0.003# micron
    elif (band == '3B'):
          xmin=510# Minimum x pixel
          xmax=1025# Maximum x pixel
          lammin=13.37# Minimum wavelength
          lammax=15.64# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.2
          rlim_arcsec=0.9
          rlimz_mic=0.007
          ps_x=0.2# arcsec
          ps_y=0.2# arcsec
          ps_z=0.003# micron
    elif (band == '3C'):
          xmin=510# Minimum x pixel
          xmax=1025# Maximum x pixel
          lammin=15.44# Minimum wavelength
          lammax=18.07# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.2
          rlim_arcsec=0.9
          rlimz_mic=0.007
          ps_x=0.2# arcsec
          ps_y=0.2# arcsec
          ps_z=0.003# micron
    elif (band == '4A'):
          xmin=8# Minimum x pixel
          xmax=509# Maximum x pixel
          lammin=17.66# Minimum wavelength
          lammax=20.93# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.3
          rlim_arcsec=1.8
          rlimz_mic=0.012
          ps_x=0.35# arcsec
          ps_y=0.35# arcsec
          ps_z=0.006# micron
    elif (band == '4B'):
          xmin=8# Minimum x pixel
          xmax=509# Maximum x pixel
          lammin=20.42# Minimum wavelength
          lammax=24.21# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.3
          rlim_arcsec=1.8
          rlimz_mic=0.012
          ps_x=0.35# arcsec
          ps_y=0.35# arcsec
          ps_z=0.006# micron
    elif (band == '4C'):
          xmin=8# Minimum x pixel
          xmax=509# Maximum x pixel
          lammin=23.89# Minimum wavelength
          lammax=28.33# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.3
          rlim_arcsec=1.8
          rlimz_mic=0.012
          ps_x=0.35# arcsec
          ps_y=0.35# arcsec
          ps_z=0.006# micron 
    else:
        print('Not implemented!')
        sys.exit(-1)

    # Note that we're assuming that the inputs have already been put
    # into surface brightness units, that divide out the specific pixel area

    print('Defining base reference coordinates')
    
    # Define 0-indexed base x and y pixel number
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    # Convert to base alpha,beta,lambda
    values=mt.xytoabl(basex,basey,band)
    basealpha=values['alpha']
    basebeta=values['beta']
    baselambda=values['lam']
    slicenum=values['slicenum']

    # Define an index that crops full detector to only pixels on a real slice
    # for this band
    index0=(np.where(baselambda > 0))[0]
    #dummy = slicenum.copy()
    # Apply the crop for the base locations
    slicenum=slicenum[index0]
    basex=basex[index0]
    basey=basey[index0]
    basebeta=basebeta[index0]
    basealpha=basealpha[index0]
    baselambda=baselambda[index0]
    
    npix=len(slicenum)
    # Convert to v2,v3 base locations
    basev2,basev3=mt.abtov2v3(basealpha,basebeta,band)

    # Create master vector of fluxes and v2,v3 locations
    master_flux=np.zeros(npix*nfiles)
    master_ra=np.zeros(npix*nfiles)
    master_dec=np.zeros(npix*nfiles)
    master_lam=np.zeros(npix*nfiles)
    master_expnum=np.zeros(npix*nfiles)
    master_dq=np.zeros(npix*nfiles,dtype=int)
    # Extra vectors for debugging
    master_detx=np.zeros(npix*nfiles) # 0-indexed
    master_dety=np.zeros(npix*nfiles) # 0-indexed
    master_v2=np.zeros(npix*nfiles)
    master_v3=np.zeros(npix*nfiles)

    ########

    print('Reading',nfiles,'inputs')
    # Loop over input files reading them into master vectors
    for i in range(0,nfiles):
        hdu = fits.open(filenames[i])
        img=hdu['SCI'].data
        dq=hdu['DQ'].data
        head0=hdu[0].header
        head1=hdu['SCI'].header

        # Pull out the fluxes identified earlier as illuminated spaxels
        thisflux=(img.reshape(-1))[index0]
        thisdq=(dq.reshape(-1))[index0]
        # Put them in the big vectors
        master_flux[i*npix:(i+1)*npix]=thisflux
        master_dq[i*npix:(i+1)*npix]=thisdq

        # Put other stuff in the big vectors for debugging
        master_v2[i*npix:(i+1)*npix]=basev2
        master_v3[i*npix:(i+1)*npix]=basev3
        master_detx[i*npix:(i+1)*npix]=basex
        master_dety[i*npix:(i+1)*npix]=basey
        master_expnum[i*npix:(i+1)*npix]=i
        master_lam[i*npix:(i+1)*npix]=baselambda

        # Transform v2,v3 coordinates to RA,dec for this exposure
        raref= head1['RA_REF']
        decref= head1['DEC_REF']
        v2ref= head1['V2_REF']
        v3ref= head1['V3_REF']
        rollref= head1['ROLL_REF']
        ra,dec,_=tt.jwst_v2v3toradec(basev2,basev3,raref=raref,decref=decref,v2ref=v2ref,v3ref=v3ref,rollref=rollref)
        # Put it in the big vectors
        master_ra[i*npix:(i+1)*npix]=ra
        master_dec[i*npix:(i+1)*npix]=dec

    #######

    # Safety case deal with 0-360 range to ensure no problems
    # around ra=0 with coordinate wraparound
    maxdiff=np.abs(np.min(master_ra)-np.max(master_ra))
    if (maxdiff > 180.):
        wrapind=np.where(master_ra > 180.)
        master_ra[wrapind]=master_ra[wrapind]-360.
        
    #medra=np.median(master_ra)

    # Declare maxima/minima of the cube range *before* doing any QA cuts for specific exposures
    lmin,lmax=lammin,lammax
    ra_min,ra_max=min(master_ra),max(master_ra)
    dec_min,dec_max=min(master_dec),max(master_dec)
    dec_ave=(dec_min+dec_max)/2.
    ra_ave=(ra_min+ra_max)/2.

    print('Wavelength limits: {} - {} micron'.format(round(lmin,2),round(lmax,2)))
    print('RA limits: {} - {} deg'.format(round(ra_min,4),round(ra_max,4)))
    print('DEC limits: {} - {} deg'.format(round(dec_min,4),round(dec_max,4)))
    
    # Eliminate any pixels with bad DQ flags (0,3,9,10,11,14,16)
    temp=(((master_dq & 2**0) == 0) & ((master_dq & 2**3) == 0) & ((master_dq & 2**9) == 0) & ((master_dq & 2**10) == 0) & ((master_dq & 2**11) == 0) & ((master_dq & 2**14) == 0) & ((master_dq & 2**16) == 0))
    good=np.where(temp == True)
    master_flux=master_flux[good]
    master_ra=master_ra[good]
    master_dec=master_dec[good]
    master_lam=master_lam[good]
    master_expnum=master_expnum[good]
    master_dq=master_dq[good]
    master_detx=master_detx[good]
    master_dety=master_dety[good]
    master_v2=master_v2[good]
    master_v3=master_v3[good]

    # Eliminate any nan fluxes
    good = ~np.isnan(master_flux)
    master_flux=master_flux[good]
    master_ra=master_ra[good]
    master_dec=master_dec[good]
    master_lam=master_lam[good]
    master_expnum=master_expnum[good]
    master_dq=master_dq[good]
    master_detx=master_detx[good]
    master_dety=master_dety[good]
    master_v2=master_v2[good]
    master_v3=master_v3[good]

    # Define the number of good total input points
    nfinal=len(master_flux)

    # Tangent plane projection to xi/eta (spatial axes)
    # Estimate min/max necessary xi/eta
    xi_min0=3600.*(ra_min-ra_ave)*np.cos(dec_ave*np.pi/180.)
    xi_max0=3600.*(ra_max-ra_ave)*np.cos(dec_ave*np.pi/180.)
    eta_min0=3600.*(dec_min-dec_ave)
    eta_max0=3600.*(dec_max-dec_ave)

    # Define integer cube sizes
    n1a=np.ceil(abs(xi_min0)/ps_x)
    n1b=np.ceil(abs(xi_max0)/ps_x)
    n2a=np.ceil(abs(eta_min0)/ps_y)
    n2b=np.ceil(abs(eta_max0)/ps_y)
    cube_xsize=int(n1a+n1b)
    cube_ysize=int(n2a+n2b)

    # Redefine xi/eta minima/maxima to exactly
    # match integer pixel boundaries
    xi_min = -n1a*ps_x - ps_x/2.
    xi_max = n1b*ps_x + ps_x/2.
    eta_min = -n2a*ps_y - ps_y/2.
    eta_max = n2b*ps_y + ps_y/2.

    print('XI limits: {} - {} deg'.format(round(xi_min,4),round(xi_max,4)))
    print('ETA limits: {} - {} deg'.format(round(eta_min,4),round(eta_max,4)))

    # Redefine x/y and ra/dec center point
    # for adopted integer pixel boundaries
    xcen=n1a
    ycen=n2a
    shiftra=(xi_min0-xi_min)/3600./np.cos(dec_ave*np.pi/180.)
    shiftdec=(eta_min0-eta_min)/3600.
    racen=ra_ave+shiftra
    decen=dec_ave+shiftdec
    
    xi=-3600.*(master_ra-racen)*np.cos(decen*np.pi/180.)
    eta=3600.*(master_dec-decen)
    cube_x=(xi-xi_min-ps_x/2.)/ps_x
    cube_y=(eta-eta_min-ps_y/2.)/ps_y
    
    # Spectral axis
    zrange=lmax-lmin
    cube_zsize=int(np.ceil(zrange/ps_z))
    lamcen=(lmax+lmin)/2.
    lamstart=lamcen-(cube_zsize/2.)*ps_z
    lamstop=lamstart+cube_zsize*ps_z
    cube_z=(master_lam-lamstart)/ps_z # Z output cube location in pixels
    wavevec=np.arange(cube_zsize)*ps_z+min(master_lam)

    dim_out = [cube_xsize,cube_ysize,cube_zsize]
    print('Cube X-Y-Z dimensions: {} spaxels'.format(dim_out))
    
    # radius of influence
    rlimx=rlim_arcsec/ps_x # in pixels
    rlimy=rlimx # in pixels
    rlimz=rlimz_mic/ps_z
    # (Gives about 1-2 spec elements at each spatial element)
    fac = 1. # default value is 1, changing this value effectively changes the final cube spatial resolution
    rlim=[rlimx*fac,rlimy*fac,rlimz] # in pixels
    print('Radius of influence in X-Y-Z direction: {} pixels'.format(rlim))

    # Exponential sigma for gaussian weighting
    expsig=expsig_arcsec/ps_x
    print('Exponential weighting sigma: {} pixels'.format(expsig))

    # Which weighting scheme?
    if (wtype == 1):
        print('Using 1/r weighting')
    if (wtype == 2):
        print('Using 1/r**2 weighting')        
    if (wtype == 3):
        print('Using exponential weighting')    
    
    # Scale correction factor is the ratio of area between an input pixel
    # (in arcsec^2) and the output pixel size in arcsec^2
    # The result means that the cube will be in calibrated units/pixel
    # scale=ps_x*ps_y/(parea)
    scale=1.0 # Output is in flux/solid angle

    # Call cube-build algorithm core
    dim_out=np.array([cube_xsize,cube_ysize,cube_zsize])
    cube=core(cube_x,cube_y,cube_z,master_flux,dim_out,expsig,rlim, scale, wtype, detx=master_detx, dety=master_dety, enum=master_expnum, detl=master_lam, psx=ps_x, psy=ps_y, psz=ps_z, **kwargs)
    # Transpose to python shape
    cube=np.transpose(cube)
    hdu=fits.PrimaryHDU(cube)

    # Construct header info
    hdu.header['CD1_1']=-ps_x/3600.
    hdu.header['CD2_2']=ps_y/3600.
    hdu.header['CD3_3']=ps_z
    hdu.header['CRPIX1']=xcen+1
    hdu.header['CRPIX2']=ycen+1
    hdu.header['CRPIX3']=1
    hdu.header['CRVAL1']=racen
    hdu.header['CRVAL2']=decen
    hdu.header['CRVAL3']=lamstart
    hdu.header['CDELT3']=ps_z
    hdu.header['CUNIT1']='deg'
    hdu.header['CUNIT2']='deg'
    hdu.header['CUNIT3']='um'
    hdu.header['CTYPE3']='WAVE'
    hdu.header['CTYPE1']='RA---TAN'
    hdu.header['CTYPE2']='DEC--TAN'
    hdu.header['BUNIT']='mJy/arcsec^2'
    hdu.header['WTYPE']=wtype
    
    # Write to file
    hdu.writeto(outcube,overwrite=True)

    
#############################

# Core of the cube-building algorithm, when all parameters have been set
# Can accept 'stopx' and 'stopy' in kwargs

def core(x,y,z,f, dim_out, expsig, rlim, scale, wtype, **kwargs):
    thisdim_out=dim_out.copy()
    # Empty output cube
    fcube = np.zeros(thisdim_out)
    maskcube=np.ones(thisdim_out)

    # Defaults for stop debug locations
    stopx,stopy,stopz=-1,-1,-1
    if ('stopx' in kwargs):
        stopx=kwargs['stopx']
    if ('stopy' in kwargs):
        stopy=kwargs['stopy']       
    if ('stopz' in kwargs):
        stopz=kwargs['stopz']

    # Defaults for debug info
    enum=np.zeros(len(f))-1
    detx=np.zeros(len(f))-1
    dety=np.zeros(len(f))-1
    detl=np.zeros(len(f))-1
    psx=-1.
    psy=-1.
    psz=-1.
    if ('detx' in kwargs):
        detx=kwargs['detx']
    if ('dety' in kwargs):
        dety=kwargs['dety']        
    if ('detl' in kwargs):
        detl=kwargs['detl']
    if ('enum' in kwargs):
        enum=kwargs['enum']
    if ('psx' in kwargs):
        psx=kwargs['psx']
    if ('psy' in kwargs):
        psy=kwargs['psy']        
    if ('psz' in kwargs):
        psz=kwargs['psz']
        
    # XYZ output pixel coordinate arrays
    arr_xcoord = np.arange(thisdim_out[0])
    arr_ycoord = np.arange(thisdim_out[1])
    arr_zcoord = np.arange(thisdim_out[2])

    # Number of total input samples
    ntot=len(f)

    # Loop over output cube building layer by layer
    for k in range(thisdim_out[2]):
        # Print a message every 5% of the loop
        if (np.mod(k,np.round(dim_out[2]/20)) == 0):
            print('Constructing cube: ',int(k/dim_out[2]*100),'% complete')

        # First pass cut: trim to only stuff within rlim of this z location
        indexk=np.where(abs(z-arr_zcoord[k]-0.5) <= rlim[2])
        nindexk = len(indexk[0])
        
        # If nothing makes the cut, then do nothing.  Otherwise build the slice
        if (nindexk > 0):
            tempx=x[indexk]
            tempy=y[indexk]
            tempz=z[indexk]
            tempf=f[indexk]
            tempenum=enum[indexk]
            temp_detx=detx[indexk]
            temp_dety=dety[indexk]
            temp_detl=detl[indexk]
            
            # Loop over output image, building the image row by row
            for j in range(thisdim_out[1]):
                # Second pass cut: trim to only stuff within rlim of this y location
                indexj=np.where(abs(tempy-arr_ycoord[j]) <= rlim[1])
                nindexj = len(indexj[0])

                # If nothing makes the cut, do nothing.  Otherwise
                # build the row
                if (nindexj > 0):
                    tempx2=tempx[indexj]
                    tempy2=tempy[indexj]
                    tempz2=tempz[indexj]
                    tempf2=tempf[indexj]
                    tempenum2=tempenum[indexj]
                    temp2_detx=temp_detx[indexj]
                    temp2_dety=temp_dety[indexj]
                    temp2_detl=temp_detl[indexj]

                    # Now do a 1d build within this slice, looping over input points
                    arr_weights=np.zeros((nindexj,thisdim_out[0]))

                    for q in range(nindexj):
                        # Don't calculate full radii for everything; if x component is too big
                        # then radius guaranteed too big.  Set default radius slightly larger than
                        # desired selection radius.  This saves a bunch of compute time.
                        arr_radius=(rlim[0]+1)*np.ones(thisdim_out[0])
                        arr_sradius=(rlim[0]+1)*np.ones(thisdim_out[0])

                        # Which output pixels are affected by input points, i.e.
                        # within rlim of this x location?
                        # Don't go outside output array boundaries
                        xmin=max(np.floor(tempx2[q]-rlim[0]).astype(int),0)
                        # Note python needs xmax 1 greater than if this was IDL...
                        xmax=min(np.ceil(tempx2[q]+rlim[0]).astype(int), thisdim_out[0]-1)+1
                        # Number of points within box
                        nbox=xmax-xmin

                        # Calculate physical spatial radius for ROI determination
                        rx=arr_xcoord[xmin:xmax]-tempx2[q]
                        ry=arr_ycoord[j]-tempy2[q]
                        rz=arr_zcoord[k]-tempz2[q]+0.5
                        arr_radius[xmin:xmax]=np.sqrt(rx**2 + np.ones(nbox)*ry**2)

                        # Determine points within the final circular ROI
                        tocalc=np.where(arr_radius <= rlim[0])
                        ncalc = len(tocalc[0])

                        # Combine normalized radii inside ROI
                        arr_sradius[xmin:xmax] = np.sqrt( (rx**2) + np.ones(nbox)* ry**2 + np.ones(nbox)* rz**2  )

                        # Ensure no divide by zero
                        if (ncalc > 0):
                            if (wtype == 0):
                                arr_weights[q,tocalc]=1.
                            elif (wtype == 1):
                                arr_weights[q,tocalc]=1./arr_sradius[tocalc]
                            elif (wtype == 2):
                                arr_weights[q,tocalc]=1./arr_sradius[tocalc]**2
                            elif (wtype == 3):
                                arr_weights[q,tocalc]=np.exp(-0.5/expsig**2*arr_sradius[tocalc]**2)
                            elif (wtype == 4):
                                arr_weights[q,tocalc]=1./arr_sradius[tocalc]**4
                        #if ((j == stopy)&(k == stopz)&(q == 6)):
                        #    pdb.set_trace()
                                
                    # Normalization matrix
                    if (nindexj == 1):
                        matr_norm=arr_weights.reshape(-1)
                    else:
                        matr_norm = np.sum(arr_weights,0)

                    # Flag where the normalization matrix is zero; there is no good datum here
                    nodata=np.where(matr_norm == 0)
                    nnodata = len(nodata[0])
                    gooddata = np.where(matr_norm != 0)
                    ngood = len(gooddata[0])
                    # Mark good locations in output mask
                    if (ngood != 0):
                        maskcube[gooddata,j,k]=0

                    # We don't want to divide by zero where there is no data# set the normalization
                    # matrix to 1 in these cases
                    if (nnodata > 0):
                        matr_norm[nodata]=1.

                    # Apply the weights to calculate the output flux in this row
                    frow=np.zeros(thisdim_out[0])

                    for q in range (nindexj):
                        alpha=arr_weights[q,:]/matr_norm
                        frow+=tempf2[q]*alpha

                    # Put the row into the final cube
                    fcube[:,j,k]=frow*scale

                # Debugging
                if ((j == stopy)&(k == stopz)):
                    temp=arr_weights[:,stopx]
                    thispix=(np.where(temp != 0))[0]
                    nthis=len(thispix)
                    thispix_weight=temp[thispix]/matr_norm[stopx]

                    # Sort according to decreasing weights
                    thispix_detx=[srt for _,srt in sorted(zip(thispix_weight,temp2_detx[thispix]),reverse=True)]
                    thispix_dety=[srt for _,srt in sorted(zip(thispix_weight,temp2_dety[thispix]),reverse=True)]
                    thispix_detl=[srt for _,srt in sorted(zip(thispix_weight,temp2_detl[thispix]),reverse=True)]
                    thispix_dx=[srt for _,srt in sorted(zip(thispix_weight,tempx2[thispix]-stopx),reverse=True)]
                    thispix_dy=[srt for _,srt in sorted(zip(thispix_weight,tempy2[thispix]-stopy),reverse=True)]
                    thispix_dz=[srt for _,srt in sorted(zip(thispix_weight,tempz2[thispix]-stopz-0.5),reverse=True)]
                    thispix_enum=[srt for _,srt in sorted(zip(thispix_weight,tempenum2[thispix]),reverse=True)]
                    thispix_flux=[srt for _,srt in sorted(zip(thispix_weight,tempf2[thispix]),reverse=True)]
                    thispix_weight.sort()
                    thispix_weight=thispix_weight[::-1]

                    print('Debug location is: ',stopx,stopy,stopz)
                    print('Final value is: ',fcube[stopx,j,k])
                    print('Cutoff xdist/ydist: ',rlim[0]*psx)
                    print('Cutff zdist: ',rlim[2]*psz)
                    print('{:>4} {:>6} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}'.format('exp','xdet','ydet','wave','xdist','ydist','zdist','rxy','flux','weight'))
                    for r in range(nthis):
                        print('{:>4} {:>6} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}'.format(thispix_enum[r].astype(int),thispix_detx[r].astype(int),thispix_dety[r].astype(int),round(thispix_detl[r],4),round(thispix_dx[r]*psx,4), round(thispix_dy[r]*psy,4), round(thispix_dz[r]*psz,4),round(np.sqrt(thispix_dx[r]**2+thispix_dy[r]**2)*psx,4),round(thispix_flux[r],4),round(thispix_weight[r],4)))
                    pdb.set_trace()
                        
    return fcube

###########

# Simple sum up detector to test Yannis approach

def simplesum(filenames,band):
    # Set the distortion solution to use
    mt.set_toolversion('cdp8b')

    nfiles=len(filenames)
    indir = os.path.dirname(os.path.realpath(filenames[0]))

    # Mid-pixel wavelength
    waveimg=mt.waveimage(band)
    waveimg_1d=waveimg.reshape(-1)
    # Bottom-pixel wavelength
    waveimg0=mt.waveimage(band,loc='bot')
    waveimg0_1d=waveimg0.reshape(-1)
    # Top-pixel wavelength
    waveimg1=mt.waveimage(band,loc='top')
    waveimg1_1d=waveimg1.reshape(-1)
    
    # Basic header information from the first file
    head0=fits.open(filenames[0])[0].header
    head1=fits.open(filenames[0])[1].header

    # simulated or ground-test data?
    if (head0['ORIGIN'] == 'MIRI European Consortium'):
        obs_type = 'mirisim'
        databand=head0['BAND']
        datadet=head0['DETECTOR'] # MIRIFUSHORT or MIRIFULONG
    else:
        print('Non-mirisim not implemented!')
        sys.exit(-1)

    # Band-specific cube-building parameters
    if (band == '1A'):
          xmin=8# Minimum x pixel
          xmax=509# Maximum x pixel
          lammin=4.89# Minimum wavelength
          lammax=5.74# Max wavelength

          # Output cube parameters
          expsig_arcsec=0.1
          rlim_arcsec=0.4
          rlimz_mic=0.0025
          ps_x=0.13# arcsec
          ps_y=0.13# arcsec
          ps_z=0.001# micron

    else:
        print('Not implemented!')
        sys.exit(-1)

    # Note that we're assuming that the inputs have already been put
    # into surface brightness units, that divide out the specific pixel area

    print('Defining base reference coordinates')
    
    # Define 0-indexed base x and y pixel number
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    # Convert to base alpha,beta,lambda
    values=mt.xytoabl(basex,basey,band)
    basealpha=values['alpha']
    basebeta=values['beta']
    baselambda=values['lam']
    slicenum=values['slicenum']

    # Define an index that crops full detector to only pixels on a real slice
    # for this band
    index0=(np.where(baselambda > 0))[0]
    #dummy = slicenum.copy()
    # Apply the crop for the base locations
    slicenum=slicenum[index0]
    basex=basex[index0]
    basey=basey[index0]
    basebeta=basebeta[index0]
    basealpha=basealpha[index0]
    baselambda=baselambda[index0]
    waveimg_1d=waveimg_1d[index0]
    waveimg0_1d=waveimg0_1d[index0]
    waveimg1_1d=waveimg1_1d[index0]
    npix=len(slicenum)
    # Convert to v2,v3 base locations
    basev2,basev3=mt.abtov2v3(basealpha,basebeta,band)

    # Create master vector of fluxes and v2,v3 locations
    master_flux=np.zeros(npix*nfiles)
    master_ra=np.zeros(npix*nfiles)
    master_dec=np.zeros(npix*nfiles)
    master_lam=np.zeros(npix*nfiles)
    master_expnum=np.zeros(npix*nfiles)
    master_dq=np.zeros(npix*nfiles,dtype=int)
    # Extra vectors for debugging
    master_detx=np.zeros(npix*nfiles) # 0-indexed
    master_dety=np.zeros(npix*nfiles) # 0-indexed
    master_v2=np.zeros(npix*nfiles)
    master_v3=np.zeros(npix*nfiles)

    ########

    print('Reading',nfiles,'inputs')
    # Loop over input files reading them into master vectors
    for i in range(0,nfiles):
        hdu = fits.open(filenames[i])
        img=hdu['SCI'].data
        dq=hdu['DQ'].data
        head0=hdu[0].header
        head1=hdu['SCI'].header

        # Pull out the fluxes identified earlier as illuminated spaxels
        thisflux=(img.reshape(-1))[index0]
        thisdq=(dq.reshape(-1))[index0]
        # Put them in the big vectors
        master_flux[i*npix:(i+1)*npix]=thisflux
        master_dq[i*npix:(i+1)*npix]=thisdq

        # Put other stuff in the big vectors for debugging
        master_v2[i*npix:(i+1)*npix]=basev2
        master_v3[i*npix:(i+1)*npix]=basev3
        master_detx[i*npix:(i+1)*npix]=basex
        master_dety[i*npix:(i+1)*npix]=basey
        master_expnum[i*npix:(i+1)*npix]=i
        master_lam[i*npix:(i+1)*npix]=baselambda

        # Transform v2,v3 coordinates to RA,dec for this exposure
        raref= head1['RA_REF']
        decref= head1['DEC_REF']
        v2ref= head1['V2_REF']
        v3ref= head1['V3_REF']
        rollref= head1['ROLL_REF']
        ra,dec,_=tt.jwst_v2v3toradec(basev2,basev3,raref=raref,decref=decref,v2ref=v2ref,v3ref=v3ref,rollref=rollref)
        # Put it in the big vectors
        master_ra[i*npix:(i+1)*npix]=ra
        master_dec[i*npix:(i+1)*npix]=dec

    #######

    lvec=np.arange(lammin,lammax,ps_z)
    nwave=len(lvec)
    spec=np.zeros(nwave)
    for ii in range(0,nwave):
        indx=np.where((waveimg0_1d <= lvec[ii])&(waveimg1_1d > lvec[ii]))
        spec[ii]=np.sum(master_flux[indx])

    return spec





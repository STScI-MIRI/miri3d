#
"""
Standalone python implementation of the modified Shepard cube-building algorithm
for the MIRI MRS.  Uses miricoord for the distortion transforms, and does not
depend on the JWST pipeline.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
01-Apr-2016  IDL version written by David Law (dlaw@stsci.edu)
08-Apr-2019  Adapted to python by Yannis Argyriou
22-May-2019  Adapted from notebook format (D. Law)
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

def cube(filenames,band,**kwargs):
    # CDP-8b distortion solution
    mt.set_toolversion('cdp6')
    
    nfiles=len(filenames)
    indir = os.path.dirname(os.path.realpath(filenames[0]))
    outdir=indir+'/pystack/'
    if os.path.exists(outdir) is False:
        os.system('mkdir {}'.format(outdir))
    outcube=outdir+'pycube.fits'
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
    if ((band == '1A')or(band == '1B')or(band == '1C')):
          pwidth=0.196# pixel size along alpha in arcsec
          swidth=0.176# slice width in arcsec
          xmin=8# Minimum x pixel
          xmax=509# Maximum x pixel

          # Output cube parameters
          rlim_arcsec=0.15# in arcseconds
          rlimz_mic=0.0025#
          ps_x=0.13# arcsec
          ps_y=0.13# arcsec
          ps_z=0.0025# micron
    else:
        print('Not implemented!')
        sys.exit(-1)


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
        
    medra=np.median(master_ra)

    # Declare maxima/minima of the cube range *before* doing any QA cuts for specific exposures
    lmin,lmax=min(master_lam),max(master_lam)
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
    xi_min=3600.*(ra_min-ra_ave)*np.cos(dec_ave*np.pi/180.)
    xi_max=3600.*(ra_max-ra_ave)*np.cos(dec_ave*np.pi/180.)
    eta_min=3600.*(dec_min-dec_ave)
    eta_max=3600.*(dec_max-dec_ave)

    # Define cube sizes
    n1a=np.ceil(abs(xi_min)/ps_x)
    n1b=np.ceil(abs(xi_max)/ps_x)
    n2a=np.ceil(abs(eta_min)/ps_y)
    n2b=np.ceil(abs(eta_max)/ps_y)
    cube_xsize=int(n1a+n1b)
    cube_ysize=int(n2a+n2b)

    # Redefine xi/eta minima/maxima to exactly
    # match pixel boundaries
    xi_min = -n1a*ps_x - ps_x/2.
    xi_max = n1b*ps_x + ps_x/2.
    eta_min = -n2a*ps_y - ps_y/2.
    eta_max = n2b*ps_y + ps_y/2.

    print('XI limits: {} - {} deg'.format(round(xi_min,4),round(xi_max,4)))
    print('ETA limits: {} - {} deg'.format(round(eta_min,4),round(eta_max,4)))

    xi=-3600.*(master_ra-ra_ave)*np.cos(dec_ave*np.pi/180.)
    eta=3600.*(master_dec-dec_ave)
    cube_x=(xi-xi_min-ps_x/2.)/ps_x
    cube_y=(eta-eta_min-ps_y/2.)/ps_y

    racen=ra_ave
    decen=dec_ave
    xcen=n1a
    ycen=n2a

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

    # Scale correction factor is the ratio of area between an input pixel
    # (in arcsec^2) and the output pixel size in arcsec^2
    # The result means that the cube will be in calibrated units/pixel
    # scale=ps_x*ps_y/(parea)
    scale=1.0 # Output is in flux/solid angle

    # Call cube-build algorithm core
    dim_out=np.array([cube_xsize,cube_ysize,cube_zsize])

    cube=core(cube_x,cube_y,cube_z,master_flux,dim_out,rlim, scale, detx=master_detx, dety=master_dety, enum=master_expnum, detl=master_lam, **kwargs)
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
    
    
    # Write to file
    hdu.writeto(outcube,overwrite=True)

    
#############################

# Core of the cube-building algorithm, when all parameters have been set
# Can accept 'stopx' and 'stopy' in kwargs

def core(x,y,z,f, dim_out, rlim, scale, **kwargs):
    thisdim_out=dim_out.copy()
    # Empty output cube
    fcube = np.zeros(thisdim_out)
    maskcube=np.ones(thisdim_out)

    # Hard coding weighting type
    wtype = 1

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
    if ('detx' in kwargs):
        detx=kwargs['detx']
    if ('dety' in kwargs):
        dety=kwargs['dety']        
    if ('detl' in kwargs):
        detl=kwargs['detl']
    if ('enum' in kwargs):
        enum=kwargs['enum']
        
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
            print('Constructing cube: ',k/dim_out[2]*100,'% complete')

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
                    thispix_detx=temp2_detx[thispix]
                    thispix_dety=temp2_dety[thispix]
                    thispix_detl=temp2_detl[thispix]
                    thispix_dx=tempx2[thispix]-stopx
                    thispix_dy=tempy2[thispix]-stopy
                    thispix_dz=tempz2[thispix]-stopz-0.5
                    thispix_enum=tempenum2[thispix]
                    thispix_flux=tempf2[thispix]
                    print('Debug location is: ',stopx,stopy,stopz)
                    print('Final value is: ',fcube[stopx,j,k])
                    print('exp xdet ydet wave xdist ydist zdist rxy flux rweight nweight')
                    for r in range(nthis):
                        # NB- HARDCODING pixel size conversions to 0.1/0.1 arcsec and 0.002 micron
                        print('{} {} {} {} {} {} {} {} {} {} {}'.format(round(thispix_enum[r],4),thispix_detx[r].astype(int),thispix_dety[r].astype(int),round(thispix_detl[r],4),round(thispix_dx[r]*0.1,4), round(thispix_dy[r]*0.1,4), round(thispix_dz[r]*0.002,4),round(np.sqrt(thispix_dx[r]**2+thispix_dy[r]**2)*0.1,4),round(thispix_flux[r],4),round(temp[thispix[r]],4),round(temp[thispix[r]]/matr_norm[stopx],4)))
                    pdb.set_trace()
                        
    return fcube
#
"""
Tool for fitting a source centroid as a function of wavelength in an MRS data cube.

Required input:
file: A 3d MRS cube produced by the JWST pipeline.

Returns: Dictionary of assorted measurements

Example:
fit_cubecen.fit('Level3_ch1-short_s3d.fits')

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
21-Nov-2022  First written
"""

from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from numpy import unravel_index
import matplotlib.pyplot as plt


from os.path import exists
import pdb

#############################

# Take a 3d cube header and turn it into a 2d image header

def hdr3dto2d(hdr):
    hdr2d=hdr.copy()
    hdr2d['NAXIS']=2
    hdr2d['WCSAXES']=2
    del hdr2d['NAXIS3'],hdr2d['CRPIX3'],hdr2d['CRVAL3'],hdr2d['CDELT3'],hdr2d['CUNIT3']
    del hdr2d['CTYPE3'],hdr2d['PC1_3'],hdr2d['PC2_3'],hdr2d['PC3_1'],hdr2d['PC3_2'],hdr2d['PC3_3']
    hdr2d.set('SIMPLE','T',before='BITPIX')
    del hdr2d['XTENSION']
    
    return hdr2d

#############################

def fit(file):
    hdu=fits.open(file)
    cube=hdu['SCI'].data
    hdr=hdu['SCI'].header

    imhdr=hdr3dto2d(hdr)
    wcsobj = WCS(imhdr)


    nz=(cube.shape)[0]
    xcen,ycen=np.zeros(nz),np.zeros(nz)
    img=np.nanmedian(cube,axis=0)
    indx=np.where(~np.isfinite(img))
    img[indx]=0.

    y, x = np.mgrid[:img.shape[0], :img.shape[1]]
    # Guess at xy peak location
    ypeak,xpeak=unravel_index(img.argmax(), img.shape)
    bounds = {"x_stddev": [0., 5.],
              "y_stddev": [0., 5.],
              "x_mean": [xpeak-5,xpeak+5],
              "y_mean": [ypeak-5,ypeak+5]}
    p_init = models.Gaussian2D(x_mean=xpeak,y_mean=ypeak,amplitude=np.nanmax(img),bounds=bounds)
    fit_p = fitting.LevMarLSQFitter()

    for ii in range(0,nz):
        img=cube[ii,:,:]
        indx=np.where(~np.isfinite(img))
        img[indx]=0.
        try:
            p = fit_p(p_init, x, y, img)
            xcen[ii]=p.x_mean.value
            ycen[ii]=p.y_mean.value
        except:
            xcen[ii]=np.nan
            ycen[ii]=np.nan

    sky = wcsobj.pixel_to_world(xcen, ycen)
    ra=sky.ra.value
    dec=sky.dec.value

    return ra,dec

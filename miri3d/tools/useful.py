#
"""
Useful python tools for working with the MIRI MRS cube building.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
21-Feb-2019  Written by David Law (dlaw@stsci.edu)
"""

from jwst import datamodels
from jwst.assign_wcs import miri
from miricoord.miricoord.mrs import mrs_pipetools as mpt
import numpy as np
import pdb

# Compute the alpha/beta/lambda corners for a given pixel on the assumption
# that traces are straight on the detector
def cornercoord(x,y,band):
    a0,b0,l0=mpt.xytoabl(x,y,band)
    a1,b1,l1=mpt.xytoabl(x-0.5,y-0.5,band)
    a2,b2,l2=mpt.xytoabl(x+0.5,y-0.5,band)
    a3,b3,l3=mpt.xytoabl(x+0.5,y+0.5,band)
    a4,b4,l4=mpt.xytoabl(x-0.5,y+0.5,band)

    amin=(a1+a4)/2.
    amax=(a2+a3)/2.
    lmin=(a1+a2)/2.
    lmax=(a3+a4)/2.
    bmin=b0-0.17721/2.
    bmax=b0+0.17721/2.

    
    
    return amin,amax,lmin,lmax,bmin,bmax

def cornercoord2(x,y,model1,model2,model3):
    a0,b0,l0=model1(x,y)
    a1,b1,l1=model1(x-0.5,y-0.5)
    a2,b2,l2=model1(x+0.5,y-0.5)
    a3,b3,l3=model1(x+0.5,y+0.5)
    a4,b4,l4=model1(x-0.5,y+0.5)

    amin=(a1+a4)/2.
    amax=(a2+a3)/2.
    lmin=(a1+a2)/2.
    lmax=(a3+a4)/2.
    bmin=b0-0.17721/2.
    bmax=b0+0.17721/2.

    avec=np.array((amin,amax,amax,amin))
    bvec=np.array((bmin,bmin,bmax,bmax))
    v2vec,v3vec=model2(avec,bvec)
    ravec,devec=model3(v2vec,v3vec)
    cc=np.array(((ravec[0],devec[0]),(ravec[1],devec[1]),(ravec[2],devec[2]),(ravec[3],devec[3])))
    
    return cc

def cornercoord_xieta(cc,ramin,demin,pixsz):
    newcc=cc.copy()
    newcc[:,0]=np.multiply(np.subtract(newcc[:,0],ramin),3600./pixsz)
    newcc[:,1]=(newcc[:,1]-demin)*3600./pixsz
    return newcc

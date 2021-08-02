#
"""
Tools for working with JWST pipeline DQ bitmask arrays.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
02-Aug-2021  First written by David Law (dlaw@stsci.edu)
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

from stcal import dqflags
from jwst import datamodels

#############################

# Print out a list of all DQ flags

def dqdef():
    datamodels.dqflags.pixel
    
    return

#############################

# Print out a list of all DQ flags set in a given bit

def dqname(value):
    dqflags.dqflags_to_mnemonics(value,mnemonic_map=datamodels.dqflags.pixel)
    
    return

#############################

# Given a DQ structure (say a 2d image) filter it to only a specific
# flag.  I.e., return it with zeros everywhere that the flag requested
# is not set, and ones where it is.
# E.g., newimage = dqimage(image,'OUTLIER')

def dqimage(image,flag):
    thebit=dqflags.interpret_bit_flags(flag,mnemonic_map=datamodels.dqflags.pixel)

    newimage=image.copy()
    indx=np.where((image & thebit) == 0)
    if (len(indx[0]) > 0):
        newimage[indx]=0
    indx=np.where((image & thebit) != 0)
    if (len(indx[0]) > 0):
        newimage[indx]=1
        
    return newimage

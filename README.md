# miri3d
Code for generation of reference files and cube building for the MIRI MRS.

## Contents:
- IDL implementations of some code for backward compatibility live in the idl/ directory
- Python code lives in the miri3d/ directory
  - /cubepar/: Code to create the pipeline cube-building parameter file (cubepars)
  - /drizzle/: Standalone implementation of a drizzle-based cube building algorithm (under development)
  - /modshep/: Standalone implementation of the Modified Shepard cube building algorithm
  - /tools/: Tools for pipeline testing

## Installation:

Python:

Python code is developed for a python 3.5 environment.  If installing via a git clone command make sure it is in a location that is on your python path.  E.g.,

export PYTHONPATH="/YourPathHere:$PYTHONPATH"

This requires that the miricoord repository be installed as well (https://github.com/STScI-MIRI/miricoord).  Some functions also rely upon having the pysiaf (https://github.com/spacetelescope/pysiaf) and JWST pipeline (https://github.com/spacetelescope/jwst) modules installed as well.

Some routines are configured to write files to a specific set of subdirectories on disk (when, e.g., generating new reference files).  The base directory for these should be set as:

setenv MIRI3D_DATA_DIR /YourDataPathHere/

If this is not set, these files will default to writing out in your current working directory.

## Dependencies:

 * Some python tools depend on the pysiaf package (https://github.com/spacetelescope/pysiaf) for interaction with the MIRI SIAF
 * Some python tools depend on the JWST calibration pipeline (https://github.com/spacetelescope/jwst)

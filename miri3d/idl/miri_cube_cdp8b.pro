;+
; NAME:
;   miri_cube
;
; PURPOSE:
;   Builds a data cube from CV ground test data or mirisim simulated
;   data. Assumes that WCS keywords have already been added to the data, either using
;   mmrs_cv_preprocess.pro (or Jane's equivalent routine) or mmrs_mirisim_preprocess
;
;   Can either run on Lvl2b slope data that has been processed by the JWST
;   pipeline or ramp-level data, in which case it will adopt a very
;   rough and hacky approach to ramp fitting.  Run in this mode by
;   specifying /rampdata
;
;   Can be run to only produce a single image slice using /imonly
;   keyword.
;
;   Can be stopped at a particular x,y,z location in the cube for
;   debugging purposes by specifying slice, stopx, and stopy
;
;   Note that the actual heavy lifting code is mmrs_cube.pro;
;   miri_cube is really the calling script.
;
; CALLING SEQUENCE:
;   miri_cube
;
; INPUTS:
;
; OPTIONAL INPUTS:
;
; OUTPUT:
;   Data cubes and slices thereof
;
; OPTIONAL OUTPUT:
;
; COMMENTS:
;   Works with CV2, CV3, and mirisim data instead of requiring
;   seperate codes for each one.
;
;   /rampdata is ramp format data
;
;   /cvint is Lvl2 data from CV testing that is neither
;   ramp nor Lvl2b format, and has no SCI extension
;
; EXAMPLES:
;
; BUGS:
;
; PROCEDURES CALLED:
;
; INTERNAL SUPPORT ROUTINES:
;
; REVISION HISTORY:
;   Early 2016    Written by David Law (dlaw@stsci.edu)
;   Oct 2016:     Changed sign on RA WCS, was wrong
;   Dec 2016:     Update for cube testing with Jane
;   07-Mar-2017   Ported to miri_cube from cv3cube and simcube.  Update to 1-index
;                 distortion mapping.
;   07-Jun-2017:  Update to use x,y in 0-indexed detector frame to
;                match python code
;   
;-
;------------------------------------------------------------------------------

pro miri_cube,band,imonly=imonly,rampdata=rampdata,cvint=cvint,stopx=stopx,stopy=stopy,stopz=stopz

; Log runtime
stime0 = systime(1)
; Memory tracking
thismem = memory()
maxmem = 0

channel=fix(strmid(band,0,1))
if (channel eq 1) then det_name='MIRIFUSHORT'
if (channel eq 2) then det_name='MIRIFUSHORT'
if (channel eq 3) then det_name='MIRIFULONG'
if (channel eq 4) then det_name='MIRIFULONG'
subband=strupcase(strmid(band,1,1));A,B, or C
if (subband eq 'A') then subband_name='SHORT'
if (subband eq 'B') then subband_name='MEDIUM'
if (subband eq 'C') then subband_name='LONG'

if ((subband ne 'A')and(subband ne 'B')and(subband ne 'C')) then begin
  print,'Subband not known!'
  exit
endif

files = dialog_pickfile( title='Read Files to Process', $
                                     filter='*.fits', $
                                     get_path=indir, $
                                     /MUST_EXIST        , $
                                     /MULTIPLE_FILES)
nfiles=n_elements(files)

; Figure out whether this is mirisim or ground test data
; by looking for the 'simulator' keyword in CREATOR keyword
hdr0=headfits(files[0])
creator=fxpar(hdr0,'CREATOR')
if (stregex(creator,'simulator',/bool)) then type='mirisim' $
else type='cv'

print,'File type: ',type

outdir=concat_dir(indir,'stack/')
if file_test(outdir,/directory) eq 0 then spawn, '\mkdir -p '+outdir
outcube=concat_dir(outdir,'cube.fits')
outcollapse=concat_dir(outdir,'collapse.fits')

; Trim the input file list to ensure that we're only using the
; appropriate band/channel
keep=intarr(nfiles)
for i=0,nfiles-1 do begin
  hdr=headfits(files[i])

  thisdet=strtrim(fxpar(hdr,'DETECTOR'),2); MIRIFUSHORT or MIRIFULONG

  ; CV data use DGAA_POS and DGAB_POS
  if (type eq 'cv') then thisband=strtrim(fxpar(hdr,'DGAA_POS'),2); Assume no crossed-setups.  SHORT,MEDIUM,LONG
  ; mirisim data use BAND
  if (type eq 'mirisim') then thisband=strtrim(fxpar(hdr,'BAND'),2)

  if ((thisband ne subband_name)or(thisdet ne det_name)) then keep[i]=1
endfor
good=where(keep ne -3,nkeep); Hack to always keep!
;stop
if (nkeep eq 0) then begin
  print,'No files match specified band/channel!'
  return
endif else begin
  print,'Discarding ',nfiles-nkeep,' files as wrong channel/band'
  files=files[good]
  nfiles=nkeep
endelse
print,'Number of good files: ',nfiles

raref=dblarr(nfiles)
decref=dblarr(nfiles)
v2ref=dblarr(nfiles)
v3ref=dblarr(nfiles)
rollref=dblarr(nfiles)

; Loop over inputs reading dither positions
for i=0,nfiles-1 do begin
  hdr=headfits(files[i])
  hdr1=headfits(files[i],exten=1)
  
  if (i eq 0) then ny=fix(sxpar(hdr1,'NAXIS2'))

  raref[i]=sxpar(hdr,'RA_REF')
  decref[i]=sxpar(hdr,'DEC_REF')
  v2ref[i]=sxpar(hdr,'V2_REF')
  v3ref[i]=sxpar(hdr,'V3_REF')
  rollref[i]=sxpar(hdr,'ROLL_REF')

  if ((v2ref[i] eq 0.)and(v3ref[i] eq 0.)and(raref[i] eq 0.)) then begin
    print,'Cannot find WCS keywords; are they present?'
  endif
endfor

; Band-specific information about pixel/slice size
; and cube-building parameters
if ((band eq '1A')or(band eq '1B')or(band eq '1C')) then begin
  pwidth=0.196; pixel size along alpha in arcsec
  swidth=0.176; slice width in arcsec
  xmin=8; Minimum x pixel
  xmax=509; Maximum x pixel

  ; Output cube parameters
  rlim_arcsec=0.15; in arcseconds
  rlimz_mic=0.0025;
  ps_x=0.13; arcsec
  ps_y=0.13; arcsec
  ps_z=0.0025; micron
endif
if ((band eq '2A')or(band eq '2B')or(band eq '2C')) then begin
  pwidth=0.196; pixel size along alpha in arcsec
  swidth=0.277; slice width in arcsec
  xmin=520; Minimum x pixel
  xmax=1020; Maximum x pixel

  ; Output cube parameters
  rlim_arcsec=0.1; in arcseconds
  rlimz_mic=0.004;
  ps_x=0.13; arcsec
  ps_y=0.13; arcsec
  ps_z=0.004; micron
endif
if ((band eq '3A')or(band eq '3B')or(band eq '3C')) then begin
  pwidth=0.245; pixel size along alpha in arcsec
  swidth=0.387; slice width in arcsec
  xmin=510; Minimum x pixel
  xmax=1025; Maximum x pixel

  ; Output cube parameters
  rlim_arcsec=0.1; in arcseconds
  rlimz_mic=0.004;
  ps_x=0.1; arcsec
  ps_y=0.1; arcsec
  ps_z=0.002; micron
endif
if ((band eq '4A')or(band eq '4B')or(band eq '4C')) then begin
  pwidth=0.273; pixel size along alpha in arcsec
  swidth=0.645; slice width in arcsec
  xmin=14; Minimum x pixel
  xmax=480; Maximum x pixel

  ; Output cube parameters
  rlim_arcsec=0.4; in arcseconds
  rlimz_mic=0.004;
  ps_x=0.2; arcsec
  ps_y=0.2; arcsec
  ps_z=0.002; micron
endif

; Ramps data are not pixel area corrected, while Lvl2b
; data are
parea=1.0
if (keyword_set(rampdata)) then parea=pwidth*swidth

nx=xmax-xmin+1

; Define 0-indexed base x and y pixel number
basex=rebin(findgen(nx)+xmin,[nx,ny])
basey=transpose(rebin(findgen(ny),[ny,nx]))

; Convert to base alpha,beta,lambda
mmrs_xytoabl_cdp8b,basex,basey,basealpha,basebeta,baselambda,band,slicenum=slicenum

; Crop to only pixels on a real slice
index0=where(slicenum gt 0,nindex0)
slicenum=slicenum[index0]
basex=basex[index0]
basey=basey[index0]
basebeta=basebeta[index0]
basealpha=basealpha[index0]
baselambda=baselambda[index0]
npix=n_elements(index0)

; Convert all alpha,beta base locations to v2,v3 base locations
mmrs_abtov2v3_cdp8b,basealpha,basebeta,basev2,basev3,band,xan=xan,yan=yan

; Create a master vector of fluxes and v2,v3 locations
master_flux=fltarr(nindex0*nfiles)
master_ra=dblarr(nindex0*nfiles)
master_dec=dblarr(nindex0*nfiles)
master_lam=fltarr(nindex0*nfiles)
master_expnum=intarr(nindex0*nfiles)
master_dq=replicate(long64(0),nindex0*nfiles)
; Extra vectors for debugging
master_detx=fltarr(nindex0*nfiles); 0-indexed
master_dety=fltarr(nindex0*nfiles); 0-indexed
master_v2=dblarr(nindex0*nfiles)
master_v3=dblarr(nindex0*nfiles)

; Loop over input files reading them into master vectors
for i=0,nfiles-1 do begin
  hdr=headfits(files[i])

  if ((type eq 'cv')and(~keyword_set(cvint))) then begin
    raw=mrdfits(files[i],'SCI',hdr1)
    dq=mrdfits(files[i],'DQ',hdr2)
    bzero=long64(fxpar(hdr2,'BZERO'))
    dq=dq+bzero
  endif
  if ((type eq 'mirisim')and(not(keyword_set(rampdata)))) then begin
    raw=mrdfits(files[i],'SCI',hdr1)
    dq=mrdfits(files[i],'DQ',hdr2)
    bzero=long64(fxpar(hdr2,'BZERO'))
    dq=dq+bzero
  endif
  if ((type eq 'mirisim')and(keyword_set(rampdata))) then begin
    raw=mrdfits(files[i],'SCI',hdr1)
    dq=mrdfits(files[i],'PIXELDQ',hdr2)
    bzero=long64(fxpar(hdr2,'BZERO'))
    dq=dq+bzero
  endif

  ; If these were ramps data, super-crude slopes conversion
  if (keyword_set(rampdata)) then begin
    ; Super-crude ramps to slopes
    ndim=size(raw,/dim)
    ; Take difference of first and last frames
    lf=raw[*,*,ndim[2]-1,*]-raw[*,*,0,*]
    ; Median across each integration to reject cosmics
    thisimg=median(lf,dimension=4)
    ;writefits,'thisimg.fits',thisimg
    ; Grow dq mask
    thisdq=ml_growmask(dq,2)
  endif else if (keyword_set(cvint)) then begin
    thisimg=(mrdfits(files[i],0,hdr1))[*,*,0]
    thisdq=replicate(0.,size(thisimg,/dim))
    hdr2=hdr1
  endif else begin
    thisimg=raw
    thisdq=dq
  endelse
  
  ; If dimensionality is not 2, something went wrong
  ndim=(size(thisimg))[0]
  if (ndim ne 2) then begin
    print,'Error: wrong input file dimensions, is this ramp data?'
    return
  endif

  ; Crop to correct 1/2 of detector
  thisflux=thisimg[xmin:xmax,*]
  thisdq=thisdq[xmin:xmax,*]
  ; Crop to only pixels on real slices
  thisflux=thisflux[index0]
  thisdq=thisdq[index0]
  master_flux[i*nindex0:(i+1)*nindex0-1]=thisflux
  master_dq[i*nindex0:(i+1)*nindex0-1]=thisdq
  ; Coordinate transform
  jwst_v2v3toradec,basev2,basev3,ra,dec,hdr=hdr1,/local

  master_ra[i*nindex0:(i+1)*nindex0-1]=ra
  master_dec[i*nindex0:(i+1)*nindex0-1]=dec
  master_lam[i*nindex0:(i+1)*nindex0-1]=baselambda
  master_expnum[i*nindex0:(i+1)*nindex0-1]=i
  master_detx[i*nindex0:(i+1)*nindex0-1]=basex
  master_dety[i*nindex0:(i+1)*nindex0-1]=basey
  master_v2[i*nindex0:(i+1)*nindex0-1]=basev2
  master_v3[i*nindex0:(i+1)*nindex0-1]=basev3
endfor

;hack
;newcen0=0.;0.005d *15. / 3600.
;newcen1=0.;0.14d / 3600.
;rad=sqrt((master_ra-newcen0)^2+(master_dec-newcen1)^2)*3600.
;newflux=1./(rad)
;master_flux=3.
;stop

; Safety case; deal with 0-360 range to ensure no problems
; around ra=0 with coordinate wraparound
maxdiff=abs(min(master_ra)-max(master_ra))
if (maxdiff gt 180.) then begin
   wrapind = where(master_ra > 180.)
   master_ra[wrapind]=master_ra[wrapind]-360.
endif

medra=median(master_ra)

; Declare maxima/minima of the cube range *before* doing any QA cuts for specific exposures
lmin=min(master_lam)
lmax=max(master_lam)
ra_min=min(master_ra)
ra_max=max(master_ra)
dec_min=min(master_dec)
dec_max=max(master_dec)
dec_ave=(dec_min+dec_max)/2.
ra_ave=(ra_min+ra_max)/2.

; Crop any pixels with bad DQ flags
badnum=long64(2)^0+long64(2)^3+long64(2)^9+long64(2)^10+long64(2)^11+long64(2)^14+long64(2)^16
bad=where((master_dq and badnum)ne 0,complement=good)
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

; Trim to eliminate any nan fluxes
index1=where(finite(master_flux) eq 1,complement=tnan)
master_flux=master_flux[index1]
master_ra=master_ra[index1]
master_dec=master_dec[index1]
master_lam=master_lam[index1]
master_expnum=master_expnum[index1]
master_dq=master_dq[index1]
master_detx=master_detx[index1]
master_dety=master_dety[index1]
master_v2=master_v2[index1]
master_v3=master_v3[index1]

; Tangent plane projection to xi/eta
xi_min=3600.*(ra_min-ra_ave)*cos(dec_ave*!PI/180.)
xi_max=3600.*(ra_max-ra_ave)*cos(dec_ave*!PI/180.)
eta_min=3600.*(dec_min-dec_ave)
eta_max=3600.*(dec_max-dec_ave)

; Define cube sizes
n1a=ceil(abs(xi_min)/ps_x)
n1b=ceil(abs(xi_max)/ps_x)
n2a=ceil(abs(eta_min)/ps_y)
n2b=ceil(abs(eta_max)/ps_y)
cube_xsize=n1a+n1b
cube_ysize=n2a+n2b

; Redefine xi/eta minima/maxima to exactly
; match pixel boundaries
xi_min = -n1a*ps_x - ps_x/2.
xi_max = n1b*ps_x + ps_x/2.
eta_min = -n2a*ps_y - ps_y/2.
eta_max = n2b*ps_y + ps_y/2.

xi=-3600.*(master_ra-ra_ave)*cos(dec_ave*!PI/180.)
eta=3600.*(master_dec-dec_ave)
cube_x=(xi-xi_min-ps_x/2.)/ps_x
cube_y=(eta-eta_min-ps_y/2.)/ps_y

racen=ra_ave
decen=dec_ave

xcen=n1a
ycen=n2a

zrange=lmax-lmin
cube_zsize=ceil(zrange/ps_z)
lamcen=(lmax+lmin)/2.
lamstart=lamcen-(cube_zsize/2.)*ps_z
lamstop=lamstart+cube_zsize*ps_z
cube_z=(master_lam-lamstart)/ps_z ; Z output cube location in pixels
wavevec=findgen(cube_zsize)*ps_z+min(master_lam)

; squash factors
xpsf_arcsec=0.3
ypsf_arcsec=0.24
zpsf_micron=1; NOT DONE
xpsf=1.;0.3;xpsf_arcsec/ps_x
ypsf=1.;0.24;ypsf_arcsec/ps_y
; Note- arbitrarily squashing x and y too much effectively
; downranks the importance of the z distance; decreasing spectral
; resolution to improve spatial resolution.

; roi
rlimx=rlim_arcsec/ps_x; in pixels
rlimy=rlimx; in pixels
rlimz=rlimz_mic/ps_z
; (Gives about 1-2 spec elements at each spatial element)
rlim=[rlimx,rlimy,rlimz]; in pixels


; Scale correction factor is the ratio of area between an input pixel
; (in arcsec^2) and the output pixel size in arcsec^2
; The result means that the cube will be in calibrated units/pixel
;scale=ps_x*ps_y/(parea)
scale=1.0 ; Output is in flux/solid angle
print,'scale=',scale

if ((keyword_set(stopz))and(keyword_set(stopx))and(keyword_set(stopy))) then $
  print,'Will stop at ',stopx,' ',stopy,' ',stopz

; Make images at specified slice
;if (~keyword_set(slice)) then slice=50
;if (~keyword_set(stopx)) then stopx=-1
;if (~keyword_set(stopy)) then stopy=-1
;if (~keyword_set(stopz)) then stopz=-1
;im=mmrs_cube(cube_x,cube_y,cube_z,master_flux,master_expnum,[cube_xsize,cube_ysize,cube_zsize],rlim,xsquash=xpsf,ysquash=ypsf,scale=scale,wtype=2,expsig=expsig,detx=master_detx,dety=master_dety,detlam=master_lam,stopx=stopx,stopy=stopy,stopz=stopz)

; Recover gaussian FWHM
;imfit=gauss2dfit(im,coeff)
;fwhmx=round(coeff[2]*2.355*ps_x*1e3)/1e3
;fwhmy=round(coeff[3]*2.355*ps_y*1e3)/1e3
;print,'Wavelength (micron): ',slice*ps_z+min(baselambda)
;print,'X FWHM (arcsec): ',fwhmx
;print,'Y FWHM (arcsec): ',fwhmy

; Write file
;mkhdr,imhdr,im
cdarr=dblarr(2,2)
cdarr[0,0]=-2.77778e-4*ps_x
cdarr[1,1]=2.77778e-4*ps_y
; Make astrometry solution, don't forget to put xcen/ycen in
; 1-indexed FITS convention
make_astr,astr,cd=cdarr,crpix=[xcen+1,ycen+1],crval=[racen,decen]
;putast,imhdr,astr
;writefits,outslice,im,imhdr

if (~keyword_set(imonly)) then begin
  cube=mmrs_cube(cube_x,cube_y,cube_z,master_flux,master_expnum,[cube_xsize,cube_ysize,cube_zsize],rlim,xsquash=xpsf,ysquash=ypsf,scale=scale,wtype=1,expsig=expsig,stopx=stopx,stopy=stopy,stopz=stopz,detx=master_detx,dety=master_dety)
  ; Write file
  mkhdr,cubehdr,cube
  putast,cubehdr,astr
  fxaddpar,cubehdr,'CD3_3',ps_z
  fxaddpar,cubehdr,'CDELT3',ps_z
  fxaddpar,cubehdr,'CRPIX3',1
  fxaddpar,cubehdr,'CTYPE3','WAVE'
  fxaddpar,cubehdr,'CRVAL3',lamstart
  fxaddpar,cubehdr,'CUNIT3','um'
  writefits,outcube,cube,cubehdr

  collapse=median(cube[*,*,30:cube_zsize-30],dimension=3)
  ; Write file
  mkhdr,collhdr,collapse
  putast,collhdr,astr
  writefits,outcollapse,collapse,collhdr
endif

; Track memory usage
thismem = memory()
maxmem = maxmem > thismem[3]
print, 'Max memory usage = ', string(maxmem/1e6,format='(f7.1)'), ' MB'

print, 'Total time for MIRI_CUBE = ', systime(1)-stime0, ' seconds', format='(a,f6.0,a)'
print, 'Successful completion of MIRI_CUBE at ' + systime()

return
end


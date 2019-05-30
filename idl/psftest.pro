pro psftest

; We'll work at 9 microns, which is 2B
; The nominal FWHM at 9 microns should be 0.349 arcsec
; according to Adrian's eqn 1
band='1A'
basefile='base_12SHORT.fits'; Reference mirisim file, processed through CALSPEC2 end that we'll use to set up file structure and headers
swidth=0.177204176691d  ; Slice width in arcsec.  Surely a better way to get this automatically, but for now it works well enough to hard code it
fwhm_input=0.349; arcsec

;Define the detector plane as a 1032x1024 grid
det=fltarr(1032,1024)
xmin=0
xmax=519
; Define 0-indexed base x and y pixel number
nx=xmax-xmin+1
ny=1024
basex=rebin(findgen(nx)+xmin,[nx,ny])
basey=transpose(rebin(findgen(ny),[ny,nx]))
; Convert to base alpha,beta,lambda for pixel left, midpoint, right
mmrs_xytoabl,basex,basey,basealpha,basebeta,baselambda,band,slicenum=slicenum
mmrs_xytoabl,basex-0.499,basey-0.499,basealphal,basebetal,baselambdal,band,slicenum=slicenuml
mmrs_xytoabl,basex+0.499,basey+0.499,basealphar,basebetar,baselambdar,band,slicenum=slicenumr
; Crop to only pixels on a real slice, where left and right edges are
; also real.  This is a bit of hack, and not quite true, but it will
; only lose us the very edge-most pixels in alpha, which is ok for our
; purposes here.
index0=where((slicenum gt 0)and(slicenuml gt 0)and(slicenumr gt 0),nindex0)
slicenum=slicenum[index0]
basex=basex[index0]
basey=basey[index0]
basebeta=basebeta[index0]
basealpha=basealpha[index0]
baselambda=baselambda[index0]
npix=n_elements(index0)
; left edges
basebetal=basebetal[index0]
basealphal=basealphal[index0]
baselambdal=baselambdal[index0]
slicenuml=slicenuml[index0]
; right edges
basebetar=basebetar[index0]
basealphar=basealphar[index0]
baselambdar=baselambdar[index0]
slicenumr=slicenumr[index0]
; Convert all alpha,beta base locations to v2,v3 base locations
mmrs_abtov2v3,basealpha,basebeta,basev2,basev3,band,xan=xan,yan=yan
mmrs_abtov2v3,basealphal,basebetal,basev2l,basev3l,band
mmrs_abtov2v3,basealphar,basebetar,basev2r,basev3r,band

; Set up the scene in RA/DEC
raobj=45.d
decobj=0.d
scene_xsize=10.;arcsec
scene_ysize=10.;arcsec
dx=0.01; arcsec/pixel
dy=dx; arcsec/pixel
scene_nx=scene_xsize/dx
scene_ny=scene_ysize/dy
scene=dblarr(scene_nx,scene_ny)
; Put a point source in the middle
xcen=fix(scene_nx/2)
ycen=fix(scene_ny/2)
scene[xcen,ycen]=1.
; Convolve with gaussian PSF
scene=filter_image(scene,fwhm_gaussian=fwhm_input/dx)
; Set up header WCS
mkhdr,scenehdr,scene
cdarr=dblarr(2,2)
cdarr[0,0]=2.77778e-4*dx
cdarr[1,1]=2.77778e-4*dy
make_astr,astr,cd=cdarr,crpix=[xcen+1,ycen+1],crval=[raobj,decobj]
putast,scenehdr,astr
writefits,'scene.fits',scene,scenehdr
; Fit a gaussian to the scene to confirm correct PSF
fit=gauss2dfit(scene,coeff)
fwhmx=round(coeff[2]*2.355*dx*1e3)/1e3
fwhmy=round(coeff[3]*2.355*dy*1e3)/1e3
print,'Nominal FWHM (arcsec): ',fwhm_input
print,'Scene X FWHM (arcsec): ',fwhmx
print,'Scene Y FWHM (arcsec): ',fwhmy

; Set up reference location for MRS
v2ref=-503.65447d
v3ref=-318.74246d

; For convenience assume roll such that alpha,beta aligned with ra/dec
; at the location of the MRS
; This only will work at dec=0 for this example
; Compute roll required for along-slice moves to be in the RA direction
a1=0.
b1=0.
a2=2.
b2=0.
mmrs_abtov2v3,a1,b1,v2_1,v3_1,band
mmrs_abtov2v3,a2,b2,v2_2,v3_2,band
jwst_v2v3toradec,v2_1,v3_1,ra_1,dec_1,v2ref=v2ref,v3ref=v3ref,raref=45.d,decref=0.d,rollref=0.d
jwst_v2v3toradec,v2_2,v3_2,ra_2,dec_2,v2ref=v2ref,v3ref=v3ref,raref=45.d,decref=0.d,rollref=0.d
dra=(ra_2-ra_1)*3600.
ddec=(dec_2-dec_1)*3600.
roll=-(atan(dra,ddec)*180/!PI-90.); degrees
; Check it
jwst_v2v3toradec,v2_1,v3_1,ra_1,dec_1,v2ref=v2ref,v3ref=v3ref,raref=45.d,decref=0.d,rollref=roll
jwst_v2v3toradec,v2_2,v3_2,ra_2,dec_2,v2ref=v2ref,v3ref=v3ref,raref=45.d,decref=0.d,rollref=roll
dra=(ra_2-ra_1)*3600.
ddec=(dec_2-dec_1)*3600.

; Set up individual dithered exposures
nexp=4
dxidl=[1.13872d,-1.02753d,1.02942d,-1.13622d]
dyidl=[-0.363763d,0.294924d,-0.291355d,0.368474d]
raref=dblarr(nexp)
decref=dblarr(nexp)
; Note that dxidl, dyidl oriented similarly to v2,v3 but with a flip
; in v2
for i=0,nexp-1 do begin
  jwst_v2v3toradec,v2ref-dxidl[i],v3ref+dyidl[i],ratemp,dectemp,v2ref=v2ref,v3ref=v3ref,raref=45.d,decref=0.d,rollref=roll
  raref[i]=ratemp
  decref[i]=dectemp
endfor

; Compute values for each exposure
allexp=fltarr(1032,1024,nexp)
for i=0,nexp-1 do begin
  print,'Doing exposure ',i
  thisexp=allexp[*,*,i]
  ; Convert central pixel locations to ra,dec for this exposure
  jwst_v2v3toradec,basev2,basev3,ra,dec,v2ref=v2ref,v3ref=v3ref,raref=raref[i],decref=decref[i],rollref=roll
  ; Left-edge pixel locations
  jwst_v2v3toradec,basev2l,basev3l,ra_left,decl,v2ref=v2ref,v3ref=v3ref,raref=raref[i],decref=decref[i],rollref=roll
  ; Right-edge pixel locations
  jwst_v2v3toradec,basev2r,basev3r,ra_right,decr,v2ref=v2ref,v3ref=v3ref,raref=raref[i],decref=decref[i],rollref=roll
  ; Top locations
  dec_upper=dec+swidth/2./3600.
  dec_lower=dec-swidth/2./3600.
   print,'npix=',npix

  ; Loop over pixels
  for j=0L,npix-1 do begin
    adxy,scenehdr,ra_left[j],dec_lower[j],x1,y1
    adxy,scenehdr,ra_right[j],dec_upper[j],x2,y2
    ; Straighten out ordering
    x_ll=round(x1 < x2)
    x_ur=round(x1 > x2)
    y_ll=round(y1 < y2)
    y_ur=round(y1 > y2)
    earea=(x_ur-x_ll+1)*(y_ur-y_ll+1)*dx*dy; Effective area in arcsec2 for the surface brightness normalization
    thisexp[basex[j],basey[j]]=total(scene[x_ll:x_ur,y_ll:y_ur])/earea
  endfor

  allexp[*,*,i]=thisexp
  ; Copy template to new file
  filename=strcompress('mock'+string(i)+'.fits',/remove_all)
  spawn,strcompress('cp '+basefile+' '+filename)
  ; Hack the header WCS info
  header=headfits(filename)
  fxaddpar,header,'V2_REF',v2ref
  fxaddpar,header,'V3_REF',v3ref
  fxaddpar,header,'RA_REF',raref[i]
  fxaddpar,header,'DEC_REF',decref[i]
  fxaddpar,header,'ROLL_REF',roll[0]
  modfits,filename,0,header
  ; Hack the Sci extension info
  sci=mrdfits(filename,'sci',header)
  fxaddpar,header,'V2_REF',v2ref
  fxaddpar,header,'V3_REF',v3ref
  fxaddpar,header,'RA_REF',raref[i]
  fxaddpar,header,'DEC_REF',decref[i]
  fxaddpar,header,'ROLL_REF',roll[0]
  modfits,filename,thisexp,header,extname='SCI'

 ; plot,ra,dec,psym=1
endfor

writefits,'allexp.fits',allexp

print,'Total Scene flux: ',total(scene)

stop



  
return
end

; REQUIRES rlim[0]=rlim[1]
; rlim is 3-element vector

; detx and dety are the actual detector x,y locations.  These are not
; needed by this code and are only for debugging, as are stopx and
; stopy which will stop at a particular cube x,y


function mmrs_cube, x, y, z, f, expnum, dim_out, rlim, scale=scale, xsquash=xsquash, ysquash=ysquash, zsquash=zsquash, maskcube=maskcube,wtype=wtype, expsig=expsig, detx=detx, dety=dety, detlam=detlam, stopx=stopx, stopy=stopy, stopz=stopz

; wtype:
; 1: 1/d weighting
; 2: 1/d^2 weighting
; 3: gaussian weighting (define expsig)
if (~keyword_set(wtype)) then wtype=1

if (~keyword_set(xsquash)) then xsquash=1.
if (~keyword_set(ysquash)) then ysquash=1.
if (~keyword_set(zsquash)) then zsquash=1.
if (~keyword_set(scale)) then scale=1.

; Add some defaults for non-necessary arguments so code doesn't crash without
if (~keyword_set(detx)) then detx=replicate(1,n_elements(x))
if (~keyword_set(dety)) then dety=replicate(1,n_elements(y))
if (~keyword_set(detlam)) then detlam=replicate(1,n_elements(z))
if (~keyword_set(stopx)) then stopx=-1
if (~keyword_set(stopy)) then stopy=-1
if (~keyword_set(stopz)) then stopz=-1

; If the slice keyword is set, make only a single slice
; and override the nominal output dimensions (ensure we
; don't pass this backwards)
thisdim_out=dim_out
;if (keyword_set(slice)) then begin
;  if ((slice ge 0)and(slice lt thisdim_out[2])) then begin
;    thisdim_out[2]=1
;    arr_zcoord=slice
;  endif
;endif

; Dimensions
ntot=n_elements(f) ; Number of total samples
all_dim=[thisdim_out,ntot] ; Output X,Y,Z size, tot sample size

; Output cubes
fcube=dblarr(thisdim_out)
maskcube=intarr(thisdim_out)
maskcube[*]=1

; XYZ output pixel coordinate arrays
arr_xcoord = dindgen(thisdim_out[0])
arr_ycoord = dindgen(thisdim_out[1])
; Don't overwrite arr_zcoord it if was already set to one slice
if (~keyword_set(arr_zcoord)) then $
  arr_zcoord = dindgen(thisdim_out[2])

;define epsilon
eps=1e-2

; Loop over output cube building the cube slice by slice
for k=0,thisdim_out[2]-1 do begin
  ; Print a message every 5% of the loop
  if (k mod fix(thisdim_out[2]/20) eq 0) then $
    print,'Constructing cube: '+strcompress(string(round(k*100./thisdim_out[2])),/remove_all)+'% complete'

  ; First pass cut: trim to only stuff within rlim of this z location
  indexk=where(abs(z-arr_zcoord[k]-0.5) le rlim[2],nindexk)

  ; If nothing makes the cut, then do nothing.  Otherwise build the slice
  if (nindexk gt 0) then begin
  tempx=x[indexk]
  tempy=y[indexk]
  tempz=z[indexk]
  tempf=f[indexk]
  tempenum=expnum[indexk]
  temp_detx=detx[indexk]
  temp_dety=dety[indexk]
  temp_detlam=detlam[indexk]

  ; QA plot
  plot,[tempx],[tempy],psym=1
  circxcen=median(tempx)
  circycen=median(tempy)
  circphi=findgen(360)/180.*!PI
  circx=circxcen+rlim[0]*cos(circphi)
  circy=circycen+rlim[1]*sin(circphi)
  oplot,circx,circy,color=250



  ; Loop over output image, building the image row by row
  for j=0,thisdim_out[1]-1 do begin
;print,j
    ; Second pass cut: trim to only stuff within rlim of this y location
    indexj=where(abs(tempy-arr_ycoord[j]) le rlim[1],nindexj)

    ; If nothing makes the cut, do nothing.  Otherwise
    ; build the row
    if (nindexj gt 0) then begin
      tempx2=tempx[indexj]
      tempy2=tempy[indexj]
      tempz2=tempz[indexj]
      tempf2=tempf[indexj]
      tempenum2=tempenum[indexj]
      temp2_detx=temp_detx[indexj]
      temp2_dety=temp_dety[indexj]
      temp2_detlam=temp_detlam[indexj]

      ; Now do a 1d build within this slice, looping over input points
      arr_weights=dblarr(thisdim_out[0],nindexj)

      for q=0L,nindexj-1 do begin
        arr_radius=replicate(rlim[0]+1,thisdim_out[0])
        arr_sradius=replicate(rlim[0]+1,thisdim_out[0])

        ; Which output pixels are affected by input points, i.e.
        ; within rlim of this x location?
        ; Don't go outside output array boundaries
        xmin=floor(tempx2[q]-rlim[0]) > 0
        xmax=ceil(tempx2[q]+rlim[0]) < (thisdim_out[0]-1)

        ; Number of points within box
        nbox=[n_elements(arr_xcoord[xmin:xmax])]

        ; Calculate physical spatial radius for ROI determination
        rx=arr_xcoord[xmin:xmax]-tempx2[q]
        ry=arr_ycoord[j]-tempy2[q]
        arr_radius[xmin:xmax]=sqrt(rx^2+replicate(ry^2,nbox))

        ; Determine points within the final circular ROI
        tocalc=where(arr_radius le rlim[0],ncalc)

        ; Squashed radii for weights
        srx=rx/xsquash
        sry=ry/ysquash
        srz=(arr_zcoord[k]-tempz2[q]+0.5)/zsquash


        ; Combine normalized radii inside ROI
        arr_sradius[xmin:xmax] = sqrt( $
          (srx^2) + $
          replicate(sry^2,nbox) + $
          replicate(srz^2,nbox)  )

        ; Ensure no divide by zero
        if (ncalc gt 0) then begin

          if (wtype eq 0) then $
            arr_weights[tocalc+q*thisdim_out[0]]=1.
          if (wtype eq 1) then $
            arr_weights[tocalc+q*thisdim_out[0]]=1./(arr_sradius[tocalc] > eps)
          if (wtype eq 2) then $
            arr_weights[tocalc+q*thisdim_out[0]]=1./(arr_sradius[tocalc]^2 > eps)
          if (wtype eq 3) then $
            arr_weights[tocalc+q*thisdim_out[0]]=exp(-0.5/expsig^2*arr_sradius[tocalc]^2)
          if (wtype eq 4) then $
            arr_weights[tocalc+q*thisdim_out[0]]=1./(arr_sradius[tocalc]^4 > eps)
        endif
        ;if ((j eq stopy)and(k eq stopz)and(q eq 6)) then stop
      endfor

      ; Normalization matrix
      if (nindexj eq 1) then matr_norm=arr_weights $
      else matr_norm = total(arr_weights,2)

      ; Flag where the normalization matrix is zero; there is no good data here
      nodata=where(matr_norm eq 0,complement=gooddata,ncomplement=ngood,nnodata)
      ; Mark good locations in output mask
      if (ngood ne 0) then maskcube[gooddata,j,k]=0

      ; We don't want to divide by zero where there is no data; set the normalization
      ; matrix to 1 in these cases
      if (nnodata gt 0) then matr_norm[nodata]=1.

      ; Apply the weights to calculate the output flux in this row
      frow=dblarr(thisdim_out[0])
      for q=0L,nindexj-1 do begin
        alpha=arr_weights[*,q]/matr_norm
        frow+=tempf2[q]*alpha
      endfor
      ; Put the row into the final cube
      fcube[*,j,k]=frow*scale
   endif

   if ((j eq stopy)and(k eq stopz)) then begin
    temp=arr_weights[stopx,*]; Cull the array weights for this x pixel
    thispix=where(temp ne 0.,nthis); Identify where weights nonzero
    if (nthis gt 0) then begin
      thispix_detx=temp2_detx[thispix]
      thispix_dety=temp2_dety[thispix]
      thispix_detlam=temp2_detlam[thispix]
      thispix_dx=tempx2[thispix]-stopx
      thispix_dy=tempy2[thispix]-stopy
      thispix_dz=tempz2[thispix]-stopz-0.5
      thispix_enum=tempenum2[thispix]
      thispix_flux=tempf2[thispix]
      print,'HARDCODED 0.1 arcsec and 0.002 micron spaxels'
      print,'Distances are in arcsec and microns'
      print,'Debug location is: ',stopx,stopy,stopz
      print,'Final value is: ',fcube[stopx,j,k]
      print,'exp xdet ydet wave xdist ydist zdist rxy flux rweight nweight'
      for r=0,nthis-1 do begin
 ; NB- HARDCODING pixel size conversions to 0.1/0.1 arcsec and 0.002 micron
        print,thispix_enum[r],fix(thispix_detx[r]),fix(thispix_dety[r]),thispix_detlam[r],thispix_dx[r]*0.1,thispix_dy[r]*0.1,thispix_dz[r]*0.002,sqrt(thispix_dx[r]^2+thispix_dy[r]^2)*0.1,thispix_flux[r],temp[thispix[r]],temp[thispix[r]]/matr_norm[stopx]
      endfor
      stop
    endif else begin
      print,'No non-zero weight contributing pixels!'
    endelse

  endif
endfor

endif
endfor


return,fcube
end

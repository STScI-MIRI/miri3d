; Program determine the Cube Defaults reference file
; THis program sets the defaults for plate scale and wavelength resolution

pro create_cubebuild_parameters_mrs

; Currently detname can have two values: MIRIFUSHORT and MIRIFULONG

instrument = 'MIRI';strupcase(instrument)

type = 'CUBEPAR'  
modelname = 'FM'
version = '07.07.00'

out_file = strcompress(instrument+'_'+modelname+'_'+type+'_'+version+'.fits' ,/remove_all)
out_filepath='/Users/sargent/MIRICLE/optimization/'+out_file
print, 'Output filename: ',out_filepath
;_______________________________________________________________________

fits_open,out_filepath,ofcb,/write
temp  = 0
mkhdr,header,temp 

get_date,date,/timetag
print,date
;add the standard header values

sxaddpar,header,'DATE',date
sxaddpar,header,'REFTYPE',type
sxaddpar,header,'DESCRIP','Default IFU Cube Sampling and ROI size '
sxaddpar,header,'PEDIGREE','GROUND'
sxaddpar,header,'TELESCOP','JWST'
sxaddpar,header,'INSTRUME',instrument

sxaddpar,header,'MODELNAM',modelname

sxaddpar,header,'DETECTOR','N/A'
;if(instrument eq 'MIRI' ) then sxaddpar,header,'DETECTOR','N/A'
sxaddpar,header,'EXP_TYPE','MIR_MRS'
sxaddpar,header,'BAND','N/A'
sxaddpar,header,'CHANNEL','N/A'

sxaddpar,header,'FILENAME',out_file
sxaddpar,header,'USEAFTER','2000-01-01T00:00:00'

sxaddpar,header,'VERSION',version
sxaddpar,header,'AUTHOR','Ben Sargent'
sxaddpar,header,'ORIGIN','STSCI'
sxaddpar,header,'FILENAME',out_file
sxaddpar,header,'HISTORY','IFU Cube defaults'
sxaddpar,header,'HISTORY','DOCUMENT: TBD'
sxaddpar,header,'HISTORY','SOFTWARE: IDL J Morrison create_cubebuild_parameters.pro'
sxaddpar,header,'HISTORY','DATA USED: Simulated Data created by Ben Sargent'
sxaddpar,header,'HISTORY','DIFFERENCES: The format of the CUBEPARS file was changed'
sxaddpar,header,'HISTORY','DIFFERENCES: per advice from David Law.  The new format'
sxaddpar,header,'HISTORY','DIFFERENCES: enables greater control over parameters '
sxaddpar,header,'HISTORY','DIFFERENCES: that define the cube-building process.'
sxaddpar,header,'HISTORY','DIFFERENCES: It also contains parameters that vary '
sxaddpar,header,'HISTORY','DIFFERENCES: smoothly as a function of wavelength to be '
sxaddpar,header,'HISTORY','DIFFERENCES: used when building multi-band and multi-'
sxaddpar,header,'HISTORY','DIFFERENCES: channel cubes.'


fits_write,ofcb,temp,header
fits_close,ofcb
;********************************************************************************
; Make a binary table to hold the values



; For MIRI the data has 12 bands defined by CH# & band
; For NIRSPEC the data as 11 bands defined by GRISM and FILTER

eps=0.01;1e-2

;fxbhmake,header,12,type
;a =['CH1','CH1','CH1','CH2','CH2','CH2','CH3','CH3','CH3','CH4','CH4','CH4']
;b =['SHORT ','MEDIUM','LONG  ','SHORT ','MEDIUM','LONG  ',$
;  'SHORT ','MEDIUM','LONG  ','SHORT ','MEDIUM','LONG  ']
chan=[1,1,1,2,2,2,3,3,3,4,4,4]
bnd=['SHORT ','MEDIUM','LONG  ','SHORT ','MEDIUM','LONG  ','SHORT ','MEDIUM','LONG  ','SHORT ','MEDIUM','LONG  ']
;[1,2,3,1,2,3,1,2,3,1,2,3];['A','B','C','A','B','C','A','B','C','A','B','C']
wmin=[4.89,5.65,6.52,7.49,8.65,9.99,11.53,13.37,15.44,17.66,20.54,23.95]
wmax=[5.75,6.64,7.66,8.78,10.14,11.71,13.48,15.63,18.05,20.92,24.40,28.45]
pscl=[0.13,0.13,0.13,0.17,0.17,0.17,0.20,0.20,0.20,0.35,0.35,0.35];[0.13,0.13,0.13,0.17,0.17,0.17,0.20,0.20,0.20,0.40,0.40,0.40]
roispat=[0.10,0.10,0.10,0.15,0.15,0.15,0.20,0.20,0.20,0.40,0.40,0.40]
wsamp=[0.001d,0.001d,0.001d,0.002d,0.002d,0.002d,0.003d,0.003d,0.003d,0.006d,0.006d,0.006d]
;[0.0025d,0.0025d,0.0030d,0.0040d,0.0045d,0.0050d,0.0060d,0.0060d,0.0080d,0.0140d,0.0140d,0.0140d]
roispec=wsamp
;[0.0025d,0.0025d,0.0030d,0.0040d,0.0045d,0.0050d,0.0060d,0.0060d,0.0080d,0.0140d,0.0140d,0.0140d]
power=[2d,2d,2d,2d,2d,2d,2d,2d,2d,2d,2d,2d]
softrad=dblarr(n_elements(chan))+(0.01d);eps^((1d)/power)



fxbhmake,header,n_elements(chan),'CUBEPAR'
fxbaddcol,1,header,chan[0],'CHANNEL'
fxbaddcol,2,header,bnd[0],'BAND'
fxbaddcol,3,header,wmin[0],'WAVEMIN',tunit ='microns'
fxbaddcol,4,header,wmax[0],'WAVEMAX',tunit='microns'
fxbaddcol,5,header,pscl[0],'SPAXELSIZE',tunit ='arcseconds'
fxbaddcol,6,header,wsamp[0],'SPECTRALSTEP',tunit='microns'

;print,header
;stop
fxbcreate,unit,out_filepath,header

fxbwritm,unit,['CHANNEL','BAND','WAVEMIN','WAVEMAX','SPAXELSIZE','SPECTRALSTEP'],$;,'ROISPATIAL','ROISPECTRAL','POWER','SOFTRAD'
  chan,bnd,wmin,wmax,pscl,wsamp;,roispat,roispec,power,softrad
fxbfinish,unit
;;;;;;;;;;;;;;;;;;;;;;;;;

fxbhmake,header2,n_elements(chan),'CUBEPAR_MSM'
fxbaddcol,1,header2,chan[0],'CHANNEL'
fxbaddcol,2,header2,bnd[0],'BAND'
;fxbaddcol,3,header,wmin[0],'WAVEMIN',tunit ='microns'
;fxbaddcol,4,header,wmax[0],'WAVEMAX',tunit='microns'
;fxbaddcol,5,header,pscl[0],'SPAXELSIZE',tunit ='arcseconds'
;fxbaddcol,6,header,wsamp[0],'SPECTRALSTEP',tunit='microns'
fxbaddcol,3,header2,roispat[0],'ROISPATIAL',tunit='arcseconds'
fxbaddcol,4,header2,roispec[0],'ROISPECTRAL',tunit='microns'
fxbaddcol,5,header2,power[0],'POWER',tunit='unitless'
fxbaddcol,6,header2,softrad[0],'SOFTRAD',tunit='unitless'
;print,header
;stop
fxbcreate,unit2,out_filepath,header2

fxbwritm,unit2,['CHANNEL','BAND','ROISPATIAL','ROISPECTRAL','POWER','SOFTRAD'],$;'WAVEMIN','WAVEMAX','SPAXELSIZE','SPECTRALSTEP',
  chan,bnd,roispat,roispec,power,softrad;wmin,wmax,pscl,wsamp,
fxbfinish,unit2

;;;;;;;;;;;;;;;;;;;;;;;;;
optpsfile='/Users/sargent/MIRICLE/optimization/opt_platescale_revised.txt'
readcol,optpsfile,lamlaw,optps,format='d,d',/silent
optroispatfile='/Users/sargent/MIRICLE/optimization/opt_roispatial_revised.txt'
readcol,optroispatfile,lamlaw,optroispat,format='d,d',/silent
optwsampfile='/Users/sargent/MIRICLE/optimization/opt_wavelengthsampling_revised.txt'
readcol,optwsampfile,lamlaw,optwsamp,format='d,d',/silent
optroispecfile='/Users/sargent/MIRICLE/optimization/opt_roispectral_revised.txt'
readcol,optroispecfile,lamlaw,optroispec,format='d,d',/silent
powercontinuous=dblarr(n_elements(lamlaw))+(2d)
softrad2ext=dblarr(n_elements(lamlaw))+(0.01d)

fxbhmake,header3,n_elements(lamlaw),'MULTICHANNEL_MSM'
fxbaddcol,1,header3,lamlaw[0],'WAVELENGTH',tunit='microns'
fxbaddcol,2,header3,optroispat[0],'ROISPATIAL',tunit='arcseconds'
fxbaddcol,3,header3,optroispec[0],'ROISPECTRAL',tunit='microns'
fxbaddcol,4,header3,powercontinuous[0],'POWER',tunit='unitless'
fxbaddcol,5,header3,softrad2ext[0],'SOFTRAD',tunit='unitless'

fxbcreate,unit3,out_filepath,header3

fxbwritm,unit3,['WAVELENGTH','ROISPATIAL','ROISPECTRAL','POWER','SOFTRAD'],$;,'SPAXELSIZE','SPECTRALSTEP'
  lamlaw,optroispat,optroispec,powercontinuous,softrad2ext;,optps,optwsamp
fxbfinish,unit3

print,'finished'
end
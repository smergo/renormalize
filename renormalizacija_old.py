import pyfits
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv
from astropy.table import Table
import warnings
from astropy.stats import sigma_clip
from numpy.random import randn
from scipy.stats import chisquare
from scipy.interpolate import splev, splrep
from scipy.interpolate import interp1d
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
from scipy.signal import argrelextrema
from astroML.fourier import PSD_continuous
from astroML.filters import min_component_filter

from helper_functions import get_spectra_dr52, spectra_normalize

warnings.filterwarnings("ignore")


#VNESI IME DATOTEKE
file_enter = 140115002101125
#170217004001220 VIJUGASTE
#160426005001122 MOLEKULSKE
#170415002001302 LOW NOISE
#150413002601363 HIGH NOISE
#170417002601152 SIROKE CRTE
#140115002101125 SIROKE CRTE

#CE MOLEKULSKE ALI PA VIJUGASTE CRTE, POTEM NASTAVI koef = 5, else = 1
koef = 1.0
#MAX IN MIN, val = 1; SAMO MAX val = 0 
val = 1
#SET sigma_cutoff
sigma_cutoff = 1.35

#NASTAVI FILTER (1 = blue, 2 = green, 3 = red, 4 = IR)
filter_value = "4"
#NASTAVI SUB-FITS (4 = normaliziran spekter)
hdr = 4
#PRIKAZI GRAF (yes or no)
plot_show = "yes"

#ISKANJE DATOTEKE
root = '/home/rok/FMF/3.stopnja/MR/Gigli/media/hdd/home2/janez/storage/HERMES_REDUCED/dr5.3/'
subfolder = 'com' 
fits_path = root + str(file_enter)[0:6] + '/standard/'+subfolder+'/' + str(file_enter) + str(filter_value) + ".fits"

def spectra(file_name,hdr):

	hdulist = pyfits.open(str(file_name))

	COUNTS = hdulist[hdr].data
	CRVAL1 = float(hdulist[hdr].header['CRVAL1'])
	CDELT1 = float(hdulist[hdr].header['CDELT1'])
	NAXIS1 = int(hdulist[hdr].header['NAXIS1'])
	
	LAMBDA = []
	for i in range(NAXIS1):
		LAMBDA.append(CRVAL1+i*CDELT1)
	
	DATE = hdulist[0].header['DATE']
	MEANRA = ("%.2f" % float(hdulist[0].header['MEANRA']))
	MEANDEC = ("%.2f" % float(hdulist[0].header['MEANDEC']))
	
	hdulist.close()

	return [LAMBDA,COUNTS,MEANRA,MEANDEC,DATE,CDELT1]

def check_spectra(hdr,plot_show):
	
	list = spectra(str(fits_path),hdr)
	LAMBDA = np.array(list[0])
	COUNTS = np.array(list[1])
	MEANRA = np.array(list[2])
	MEANDEC = np.array(list[3])
	DATE = np.array(list[4])
	CDELT1 = np.array(list[5])
	
	counts_norm_fit = spectra_normalize(LAMBDA-np.median(LAMBDA), COUNTS, steps=20, sigma_low=2.5, sigma_high=3, order=6, n_min_perc=5., return_fit=True, func='cheb')
	
	#EXTREMI PO SIGMA CLIPPINGU
	#COUNTS_CLIP = COUNTS
	COUNTS_CLIP = sigma_clip(COUNTS, sigma_lower = 3.0, sigma_upper = 5.0, iters=5)
	
	if val == 0:
		extrema = argrelextrema(COUNTS_CLIP, np.greater)
	else:
		aa = argrelextrema(COUNTS_CLIP, np.greater)
		bb = argrelextrema(COUNTS_CLIP, np.less)
		extrema = np.sort(np.hstack((aa,bb)))
	
	#TUPPLE -> ARRAY
	xtrm = []
	index = []
	for i in range(len(extrema[0])):
		xtrm.append(LAMBDA[extrema[0][i]])
		index.append(extrema[0][i])
	
	#RAZMIK MED EKSTREMI
	j = 0
	count = 0
	DELTA = []
	while j < len(xtrm)-1:
		delta = xtrm[j+1]-xtrm[j]
		DELTA.append(delta)
		count += delta
		j += 1
	
	#FOR ZANKA ZA ISKANJE UGODNIH PARAMETROV
	LEN_VORTEX = []
	LOW_LIMIT = []
	HIGH_LIMIT = []
	for r1 in np.arange(1.1,1.5,0.1):
		for r2 in np.arange(1.5,2.5,0.1):
			#INITIAL low value = 1.5*mediana, high value = 2.0*mediana
			xtrm_f_test = []
			index_f_test = []
			for k in range(len(DELTA)-1):
				if (xtrm[k+1]-xtrm[k] >= r1*np.median(DELTA)) and (xtrm[k+1]-xtrm[k] <= r2*np.median(DELTA)):
					xtrm_f_test.append(xtrm[k+1])
					index_f_test.append(index[k+1])
			
			print "ALL >> Stevilo vozlisc = "+str(len(xtrm_f_test))+", low = "+str(r1)+", high = "+str(r2)
			LEN_VORTEX.append(np.abs(len(xtrm_f_test)-200)) #razlika od opt. stevila 150
			LOW_LIMIT.append(r1)
			HIGH_LIMIT.append(r2)
	
	#ISKANJE NAJUGODNEJSEGA STEVILA VOZLISC		
	low = LOW_LIMIT[LEN_VORTEX.index(np.min(LEN_VORTEX))]
	high = HIGH_LIMIT[LEN_VORTEX.index(np.min(LEN_VORTEX))]
	
	#VZAMEM NAJBOLJSI PRIMER
	xtrm_f = []
	index_f = []
	for k in range(len(DELTA)-1):
		if (xtrm[k+1]-xtrm[k] >= low*np.median(DELTA)) and (xtrm[k+1]-xtrm[k] <= high*np.median(DELTA)):
			xtrm_f.append(xtrm[k+1])
			index_f.append(index[k+1])
	print "------------------------------------------------------------"
	print "BEST >> Stevilo vozlisc = "+str(len(xtrm_f))+", low = "+str(r1)+", high = "+str(r2)
	#PRVI POLINOMSKI FIT
	z = np.polyfit(xtrm_f-np.median(xtrm_f),COUNTS[index_f],24)
	p = np.poly1d(z)
	
	#ODSTOPANJE VOZLISC OD POLINOMA
	r = 0
	count_r = 0
	DELTA_r = []
	INDEX_r = []
	COUNTS_r = []
	LAMBDA_r = []
	while r < len(COUNTS[index_f]):
		delta = COUNTS[index_f][r] - (p(xtrm_f[r]-np.median(xtrm_f)))
		DELTA_r.append(delta)
		INDEX_r.append(index_f[r])
		LAMBDA_r.append(xtrm_f[r])
		COUNTS_r.append(COUNTS[index_f][r])
		r += 1
	print "Mediana razlike = "+str(np.median(DELTA_r))
	print "Povprecje razlike = "+str(np.average(DELTA_r))
	print "Standardna deviacija razlike = "+str(np.std(DELTA_r))
	
	xtrm_ffs = []
	ytrm_ffs = []
	for s in range(len(COUNTS_r)):
		if np.abs(DELTA_r[s]) - np.average(DELTA_r) <= np.std(DELTA_r):
			xtrm_ffs.append(LAMBDA_r[s])
			ytrm_ffs.append(COUNTS_r[s])
	
	#PRILAGODITEV ROBOV
	xtrm_ff = []
	ytrm_ff = []
	 
	for v in range(len(xtrm_ffs)):
		if np.abs(ytrm_ffs[v] - np.median(ytrm_ffs)) <= koef*sigma_cutoff*np.std(ytrm_ffs):
			ytrm_ff.append(ytrm_ffs[v])
			xtrm_ff.append(xtrm_ffs[v])
	xtrm_ff[0] = LAMBDA[0]
	ytrm_ff[0] = np.median(COUNTS[15:100])
	xtrm_ff[-1] = LAMBDA[len(LAMBDA)-1]
	ytrm_ff[-1] = np.median(COUNTS[-100:-15])
	
	print "FINAL >> Stevilo vozlisc = "+str(len(xtrm_ff))	
	
	#DRUGI POLINOMSKI FIT
	z1 = np.polyfit(xtrm_ff-np.median(xtrm_ff),ytrm_ff,9)
	p1 = np.poly1d(z1)		
	
	#RENORMALIZACIJA
	COUNTS_RENORM = []
	for l in range(len(COUNTS)):
		COUNTS_RENORM.append(COUNTS[l]/(p1(LAMBDA[l]-np.median(xtrm_ff))))
	
	'''
	#KONVOLUCIJA - preverim konvolucijo med poly1 in poly2
	convolution = np.convolve(p(LAMBDA-np.median(xtrm_f)),p1(LAMBDA-np.median(xtrm_ff)),mode='full')
	convolution = convolution/np.max(convolution) 
	wavelength = np.linspace(-1,1,len(convolution)) 
	print len(wavelength) 
	
	box = []
	for i in range(len(wavelength)):
		if wavelength[i] < -0.5 or wavelength[i] > 0.5:
			box.append(0.0)
		else:
			box.append(1.0)
	convolution_theo = np.convolve(box,box,mode='same')
	convolution_theo = convolution_theo/np.max(convolution_theo) 
	difference = convolution_theo/convolution	
	'''			
	#PLOT
	if plot_show == "yes":
		
		#ZGORNJI GRAF
		fig = plt.figure(figsize=(5, 3.75))
		fig.subplots_adjust(hspace=0.25)
		ax = fig.add_subplot(211)
		ax.set_xlabel('$\lambda (\AA)$', fontsize=14, color='black')
		ax.set_ylabel('Normalized Counts', fontsize=14, color='black')
		ax.plot(LAMBDA,COUNTS,c='gray',linewidth=1,label="Original")
		ax.plot(LAMBDA,p(LAMBDA-np.median(xtrm_f)),'g-',linewidth=1, label="Poly fit")
		ax.plot(xtrm_ff,ytrm_ff,'bo',label="Extrema sigma filtered")
		ax.plot(LAMBDA,p1(LAMBDA-np.median(xtrm_ff)),'r-',linewidth=1, label="New poly fit")
		ax.plot(LAMBDA,counts_norm_fit ,'m-',linewidth=1, label="Klemen")
		ax.legend()
		
		#SPODNJI GRAF
		ax = fig.add_subplot(212)
		ax.axhline(1,color='red',ls="-")
		ax.set_xlabel('$\lambda (\AA)$', fontsize=14, color='black')
		ax.set_ylabel('Normalized Counts', fontsize=14, color='black')
		ax.plot(LAMBDA,COUNTS_RENORM,c='gray',linewidth=1, label="Renormalized")
		#ax.plot(wavelength,convolution,c='gray',linewidth=1, label="Convolution")
		#ax.plot(wavelength,convolution_theo,'k--', label="Theoretical")
		#ax.plot(wavelength,difference,'b-', label="Difference")
		ax.legend(loc=3)
		plt.show()
			
	return

file_name = str(file_enter)+str(filter_value)+".fits"

print "Spekter = "+str(file_enter)+".fits"
if int(filter_value) == 1:
	print "Filter = blue"
if int(filter_value) == 2:
	print "Filter = green"
if int(filter_value) == 3:
	print "Filter = red"
if int(filter_value) == 4:
	print "Filter = IR"
print "hdr = "+str(hdr)
print "------------------------------------------------------------"

check_spectra(hdr,plot_show)


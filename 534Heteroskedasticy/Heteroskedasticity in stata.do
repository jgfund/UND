 *************
 ****Do file to test for heteroskedasticity using the White (1980) test
 ****Also show how to use WLS/GLS to correct for heteroskedasticity
 
 *use "Z:\Home\TEACHING\Econ 306\5. Quantitative Modelling\homepriceV10.dta" ,clear
 use "C:\Users\jfras\OneDrive\UND\534AppliedEcon\Datasets\homepriceV10.dta" ,clear
 *White Test for heteroskedasticity
 regress price lot floor bed
 estimates store t1c1
 predict uhat, resid
 gen uhat_2 = uhat*uhat
 
 gen lot_floor = lot*floor
 gen lot_bed = lot*bed
 gen floor_bed  = floor*bed
 gen lot_2 = lot*lot
 gen floor_2 = floor*floor
 gen bed_2 = bed*bed
 
 regress uhat_2  lot floor bed lot_floor lot_bed floor_bed lot_2 floor_2 bed_2 
 *e(N) = number of observations from the regression
 *e(r2) = r squared from the regression
 di chi2tail(9, e(r2)*e(N))
 
  
 *White Test for heteroskedasticity (Quick way)
 regress price lot floor bed
 whitetst
 
 *Robust SE
 regress price lot floor bed, robust
 estimates store t1c2
 
 
 *esttab t1c1 t1c2 using "Z:\Home\TEACHING\Econ 534\Ch 5\home_priceA.csv",  se ar2 b(%10.4f) star(* 0.1 ** 0.05 *** 0.01) nogaps replace
esttab t1c1 t1c2 using "C:\Users\jfras\OneDrive\UND\534AppliedEcon\MultipleRegression\P-home_priceA.csv",  se ar2 b(%10.4f) star(* 0.1 ** 0.05 *** 0.01) nogaps replace

*BOSTON Dataset 
use  "C:\Users\jfras\OneDrive\UND\534AppliedEcon\Datasets\hprice1.dta" , clear

regress price lotsize sqrft bdrms
estimates store t1c3
whitetst

*Using the logarithm
regress lprice llotsize lsqrft bdrms
whitetst

regress price lotsize sqrft bdrms, robust
estimates store t1c4

*esttab t1c3 t1c4 using "Z:\Home\TEACHING\Econ 534\Ch 5\home_priceB.csv",  se ar2 b(%10.4f) star(* 0.1 ** 0.05 *** 0.01) nogaps replace
esttab t1c3 t1c4 using "C:\Users\jfras\OneDrive\UND\534AppliedEcon\MultipleRegression\P-home_priceA.csv",  se ar2 b(%10.4f) star(* 0.1 ** 0.05 *** 0.01) nogaps replace
**APPLYING GLS 
regress price lotsize sqrft bdrms
predict uhat, resid
gen uhat_2 = uhat*uhat

regress uhat_2 lotsize sqrft 
predict uhat2_fitted, xb

gen price_t = price /sqrt(uhat2_fitted)
gen lotsize_t = lotsize / sqrt(uhat2_fitted)
gen sqrft_t = sqrft /sqrt(uhat2_fitted)
gen bdrms_t = bdrms / sqrt(uhat2_fitted)
gen const_t = 1/sqrt(uhat2_fitted) 

*Transform by 1/sqrt(h)
regress  price_t lotsize_t sqrft_t bdrms_t const_t , noconst
 

 **WLS  (weight by 1/h)
 regress price lotsize sqrft bdrms [aw = 1/uhat2_fitted]

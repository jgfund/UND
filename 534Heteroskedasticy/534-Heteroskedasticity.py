#!/usr/bin/env python
# coding: utf-8

# In[1]:


#From econ 534 Heteroskedasticity lesson
#from Multiple Regression Section
#test for heteroskedasticity using the White (1980) test
#show how to use WLS/GLS to correct for hetereoskedasticity
#White test for heteroskedasticity
#call in homepriceV10.dta data, .dta is a STATA file


# In[2]:


#let's prepare a new dataframe
import pandas as pd
hpv10data = 'C:/Users/jfras/OneDrive/UND/534AppliedEcon/Datasets/homepriceV10.dta'
#make sure you preserve data types, otherwise it all comes in as strings
housedata = pd.read_stata(hpv10data, preserve_dtypes=True)
df1 = pd.DataFrame(housedata)
df1['status']=df1['status'].astype('string')
df1['school']=df1['school'].astype('string')
#take a look at the data frame
df1.head()


# In[3]:


print('Check your data Datatypes after import')
print(df1.dtypes)
#if you see any ojbect data types, go back up to prior cell and change them to strings
#look in the prior cell where 'status' and 'school' were changed to strings


# In[4]:


#some summary stats
df1.describe()


# In[5]:


#take a look at the data frame shape
df1.shape
#(rows,columns)


# In[6]:


#descriptive analysis by graphing
#using pyplot from matplotlib
import numpy as np
import matplotlib.pyplot
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

#known issue with having import statement in same cell as graph code,
#graph won't work until you executive block twice

#https://github.com/jupyter/notebook/issues/3691
#https://github.com/ipython/ipython/pull/11916
#code from pull 11916
#import ipykernel.pylab.backend_inline
#matplotlib.rcParams['backend'] = backend
#matplotlib.pyplot.switch_backend(backend)
#plt.switch_backend(backend)
#plt.show._needmain = False


# In[7]:


#some options...
#plt.ion()
#plt.figure(figsize=(15,10))
plt.tight_layout()
#chet
#style is the third optional argument which is the color and line type
df1.plot(x='floor', y='price', style='o')
plt.title('Square Feet vs Price')
plt.xlabel('Floor')  
plt.ylabel('Price')
plt.pause(.10)
plt.show(block=False)

#legend disappears with this method below
##plt.plot('floor', 'price', 'o', data=df)
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html


# In[8]:


#check distribution of price using seaborn
import seaborn as seabornInstance 
#plt.figure(figsize=(15,10))
#plt.tight_layout()
seabornInstance.distplot(df1['price'])


# In[9]:


#setup 3 variables as a dataframe of independent variables for later use
flbdata = pd.read_stata(hpv10data,columns=['floor', 'lot', 'bed'],preserve_dtypes=True).values
#X = pd.read_stata(hpv10data, preserve_dtypes=True)
X = pd.DataFrame(flbdata)
X.columns =['floor','lot','bed']
#a constant must be added for the het_white test to work
X['constant']=1
X.shape
#print(X)
X.columns.values
X.head()
#print(X.dtypes)


# In[10]:


#setup y as the dependent variable dataframe for later use
pricedata = df1['price'].values.reshape(-1,1)
y = pd.DataFrame(pricedata)
y.columns =['price']
y.shape
y.head()
print(y.dtypes)


# In[11]:


#create the regression using stats models
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
#regress price on lot, floor, bed
expr1 = 'y ~ lot + floor + bed'
lm1 = ols(expr1, df1).fit()
print(lm1.summary())


#can print specific parts of the results
#print(regr.intercept_)
#print(regr.coef_)

#could have use sklean as below but, statsmodels anova table looks nicer
#from sklearn.linear_model import LinearRegression
#regress price on lot, floor, bed
#regr = LinearRegression()
#regr.fit(X, y)


# In[12]:


#you can use these commands to print the attributes of the regeression, similair to STATA's "e"
#print("summary()\n",lm1.summary())
#print("pvalues()\n",lm1.pvalues)
#print("tvalues()\n",lm1.tvalues)
print("rsquared()\n",lm1.rsquared)
#print("rsquared_adj()\n",lm2.rsquared_adj)
print("parameters()\n",lm1.params)

#https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.html#statsmodels.regression.linear_model.OLSResults


# In[13]:


#https://www.statsmodels.org/v0.10.2/examples/notebooks/generated/predict.html


# In[14]:


print("residuals()\n",lm1.resid)


# In[15]:


print("predicted value()\n",lm1.fittedvalues)


# In[16]:


#add predicted and residual value to dataframe
df1['yhat'] = lm1.fittedvalues
df1['uhat'] = lm1.resid
#df1.shape
#if you want to look a visual of dataframe at this point
df1.head()


# In[17]:


# calculate residual (uhat) for plotting
#uhat= df1.price-df1.yhat
#print(uhat)
#uhat.shape


# In[18]:


#first create the predicted value of y for plotting
#yhat = lm1.predict()
#print(yhat)


# In[19]:


#visualize price on x axes and predicted price on y axes
plt.plot(df1.price, df1.yhat, 'o', color='black')
plt.xlabel('Actual Price')  
plt.ylabel('Predicted value')
plt.title("Actual Price vs Predicted Price")


# In[20]:


#LONG METHOD FOR WHITE TEST
#now do the white test for heteroskedasticity showing the work done by the library function het_white behind the scenes


# In[21]:


#prep some interative variables for white test


# In[22]:


#put residual in a data frame for future use
#df9a = pd.DataFrame(uhat)
#df9a.columns =['uhat']

#df9a.head()
#df9a.shape
#df9a.columns.values
#print(df9a.dtypes)


# In[23]:


#residual plot
#df9a is the residual from cell above
plt.plot(df1.yhat,df1.uhat, 'o', color='darkblue')
plt.title("Residual Plot")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")


# In[24]:


#to do the long method of the white test(show how it works)
#create some interaction variables and add them to the data frame
#create some interaction variables and add them to the data frame
df1['uhat_2'] = df1.uhat * df1.uhat
df1['lot_floor'] = df1.lot * df1.floor
df1['lot_bed'] = df1.lot * df1.bed
df1['floor_bed'] = df1.floor * df1.bed
df1['lot_2'] = df1.lot * df1.lot
df1['floor_2'] = df1.floor * df1.floor
df1['bed_2'] = df1.bed * df1.bed
#check the dataframe, previously was 21 columns, should have added 7 columns for a total of 28
df1.shape
#df.tail()


# In[25]:


df1.head()


# In[26]:


#if you make a mistake you can drop a variable with for example df.drop([uhat_2,axis=1])
#axis=1 means column not a row


# In[27]:


#using statsmodels
#and using these interactive variables and the residual uhat from above
#regress residual squared on 'floor', 'lot', 'bed','lot_floor','lot_bed','floor_bed','lot_2','floor_2','bed_2'
expr2 = 'uhat_2 ~ lot + floor + bed + lot_floor + lot_bed + floor_bed + lot_2 + floor_2 + bed_2'
lm2 = ols(expr2, df1).fit()
print(lm2.summary())


# In[28]:


#you can use these commands to print the attributes of the regeression, similair to stata's "e"
#print("summary()\n",lm2.summary())
#print("pvalues()\n",lm2.pvalues)
#print("tvalues()\n",lm2.tvalues)
print("rsquared()\n",lm2.rsquared)
#print("rsquared_adj()\n",lm2.rsquared_adj)
print("parameters()\n",lm2.params)


# In[29]:


#you can declare variables from regression attributes for later use
#R squared
rsquaredlm2 = lm2.rsquared
print (rsquaredlm2)
#degrees of freedom
degfreedmodlm2 = lm2.df_model
print(degfreedmodlm2)
#degrees of freedom for the residutal
degfreedreslm2 = lm2.df_resid
print(degfreedreslm2)
#number of observations for regression
numobslm2 = lm2.nobs
print(numobslm2)


# In[30]:


#try to emulate STATA's di chi2tail(9, e(r2)*e(N)), #di is display
#--Description: the reverse cumulative (upper tail or survivor) 
#--X2 distribution with df degrees of
#--freedom; 1 if x < 0
#--chi2tail(df,x) = 1 - chi2(df,x)


# In[31]:


#comparison for upcoming chi squared test
from scipy.stats import chi2
#scipy.stats.chi2(*args, **kwds) = <scipy.stats._continuous_distns.chi2_gen object>
#find the critical value for a 95% confidence level
value = chi2.ppf(0.95, degfreedmodlm2)
print("crifical value for 95% conficence level with 9 degrees of freedom \n",value)

#confirm with cdf
p= chi2.cdf(value, degfreedmodlm2)
print(p)

#scipy.stats.chi2(*args, **kwds) = <scipy.stats._continuous_distns.chi2_gen object>
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.chi2.html


# In[32]:


arg2=(lm2.rsquared * lm2.nobs)
value2 = chi2.cdf(arg2, degfreedmodlm2)
value3 = (1-(chi2.cdf(arg2, degfreedmodlm2)))
print(value2)
print(value3)


# In[33]:


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html#scipy.stats.chisquare
#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-25-chi.html    


# In[34]:


#some white test instructions
#null for white test is homoskedasticity
#faling to reject null means homoskedasticity

#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-25-chi.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power_divergence.html#scipy.stats.power_divergence
#https://www.programiz.com/python-programming/tuple
#https://mgimond.github.io/Stats-in-R/ChiSquare_test.html


# In[35]:


#quick white test
#returns
#-Lagrange multiplier stat
#-p-value of Langrange multiplier test
#-F-Statistic of the hypothesis that error variance does not depend on x
#-p-value of the F statistic

# X dataframe is from cell 9
# y dataframe is from cell 10
#df9a dataframe is from cell 18
#df1 dataframe is updated in cell 19

from statsmodels.stats.diagnostic import het_white
white_test1 = het_white(lm1.resid,X)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic','F-Test p-value']
print(dict(zip(labels, white_test1)))

#https://medium.com/@remycanario17/tests-for-heteroskedasticity-in-python-208a0fdb04ab
#https://www.statsmodels.org/v0.10.1/generated/statsmodels.stats.diagnostic.het_white.html
#https://medium.com/keita-starts-data-science/heteroskedasticity-in-linear-regressions-and-python-16eb57eaa09


# In[36]:


#quick breuschpagan test
from statsmodels.stats.diagnostic import het_breuschpagan
pg_test1 = het_breuschpagan(lm1.resid,X)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic','F-Test p-value']
print(dict(zip(labels, pg_test1)))


# In[37]:


#robust standard errors
#regress price on lot, floor, bed
expr3 = 'y ~ lot + floor + bed'
lm3 = ols(expr3, df1).fit(cov_type='HC1')
print(lm3.summary())


# In[38]:


#now we'll collect some of the above regressions and print in an excel table
#import the summary_col from the statsmodels library
from statsmodels.iolib.summary2 import summary_col
#example,dictionary of values to be called, .2f is a float with two decimals, d rounds to integer
info_dict1={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

results_table1 = summary_col(results=[lm1,lm2,lm3],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['Model 1',
                                         'Model 2',
                                         'Model 3'
                                         ],
                            info_dict=info_dict1,
                            regressor_order=['Intercept',
                                             'lot',
                                             'floor',
                                             'bed'])

results_table1.add_title('Table 1 - OLS Regressions')

print(results_table1)


# In[39]:


#pandas built-in csv module
import csv
#three file types supported
#results_text1 = results_table.as_latex()
#results_text1 = results_table.as_html()
results_text1 = results_table1.as_text()
#the r converts a normal string to a raw string
#'w' is write mode, 'r' is read mode
resultFile1 = open(r'C:\Users\jfras\OneDrive\DataScience\Python\table1.csv','w',newline='')
resultFile1.write(results_text1)
resultFile1.close()
#this produces a csv with all columns in one, look at the next block of code

#pandas function to read and write the csv doesn't do anything except change the extension
read_file = pd.read_csv (r'C:\Users\jfras\OneDrive\DataScience\Python\table1.csv',sep='\t')
read_file.to_excel (r'C:\Users\jfras\OneDrive\DataScience\Python\table1x.xlsx', index = None, header=True)

#https://realpython.com/python-csv/#writing-csv-files-with-csv
#https://realpython.com/python-csv/#writing-csv-file-from-a-dictionary-with-csv


# In[40]:


### HERE WE START USING A NEW DATA file, has log of some fields


# In[41]:


hprice1data = 'C:/Users/jfras/OneDrive/UND/534AppliedEcon/Datasets/hprice1.dta'
#make sure you preserve data types, otherwise it all comes in as strings
housedata2 = pd.read_stata(hprice1data, preserve_dtypes=True)
df20 = pd.DataFrame(housedata2)
#take a look at the data frame
df20.head()
#df20.shape


# In[42]:


print('Check your data Datatypes after import')
print(df20.dtypes)
#if you see any ojbect data types, go back up to prior cell and change them to strings
#look in the prior cell where 'status' and 'school' were changed to strings


# In[43]:


#regress price on lotsize, square feet, bedrooms
#expr20 = 'df20.price ~ df20.lotsize + df20.sqrft + df20.bdrms'
expr20 = 'price ~ lotsize + sqrft + bdrms'
lm20 = ols(expr20, df20).fit()
print(lm20.summary())


# In[44]:


print("residuals()\n",lm20.resid)


# In[45]:


print("predicted value()\n",lm20.fittedvalues)


# In[46]:


#from hprice1.dta
#setup 3 variables as a dataframe of independent variables for later use
exogdata21 = pd.read_stata(hprice1data,columns=['lotsize','sqrft','bdrms'],preserve_dtypes=True).values
#X = pd.read_stata(hpv10data, preserve_dtypes=True)
X21 = pd.DataFrame(exogdata21)
X21.columns =['lotsize','sqrft','bdrms']
X21["constant"]=1
X21.shape
#print(X)
X21.columns.values
#X21.head()
print(X21.dtypes)


# In[47]:


white_test2 = het_white(lm20.resid,X21)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic','F-Test p-value']
print(dict(zip(labels, white_test2)))


# In[48]:


#now do a regression using logs


# In[49]:


#regress price on lotsize, square feet, bedrooms
#expr30 = 'df20.lprice ~ df20.llotsize + df20.lsqrft + df20.bdrms'
expr30 = 'lprice ~ df20.llotsize + df20.lsqrft + bdrms'
lm30 = ols(expr30, df20).fit()
print(lm30.summary())


# In[50]:


#from hprice1.dta
#setup 3 variables as a dataframe of independent variables for later use
exogdata31 = pd.read_stata(hprice1data,columns=['llotsize','lsqrft','bdrms'],preserve_dtypes=True).values
#X = pd.read_stata(hpv10data, preserve_dtypes=True)
X31 = pd.DataFrame(exogdata31)
X31.columns =['log lotsize','log sqrft','bdrms']
X31["constant"]=1
X31.shape
#print(X)
X31.columns.values
#X31.head()
print(X31.dtypes)


# In[51]:


white_test3 = het_white(lm30.resid,X31)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic','F-Test p-value']
print(dict(zip(labels, white_test3)))


# In[52]:


#now do a regression using robust standard errors
#regress price on lotsize, square feet, bedrooms
#expr40 = 'df20.price ~ df20.lotsize + df20.sqrft + df20.bdrms'
expr40 = 'price ~ lotsize + sqrft + bdrms'
lm40 = ols(expr40, df20).fit(cov_type='HC1')
print(lm40.summary())

#HC1 applies a degrees of freedom-based correction, (n−1)/(n−k) 
#where n is the number of observations and k is the number of 
#explanatory or predictor variables in the model.
#https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html#statsmodels.regression.linear_model.RegressionResults.get_robustcov_results


# In[53]:


### Applying GLS ###


# In[54]:


#going back to the df20 dataframe (from the hprice1.dta dataset)
#we repeat the lm1 model residuals and predicted values
print("residuals()\n",lm20.resid)
print("predicted value()\n",lm20.fittedvalues)


# In[55]:


#back to dataframe20 (df20)
df20.head()


# In[56]:


#don't need to rerun model as we do in STATA


# In[57]:


#add predicted value, residual and residual squared to df20
df20['yhat'] = lm20.fittedvalues
df20['uhat'] = lm20.resid
df20['uhat_2'] = df20.uhat * df20.uhat
#df20.shape
#if you want to look a visual of dataframe at this point
df20.head()


# In[58]:


#regress price on lotsize, square feet, bedrooms
#expr50 = 'df20.uhat_2 ~ df20.lotsize + df20.sqrft'
expr50 = 'uhat_2 ~ lotsize + sqrft'
lm50 = ols(expr50, df20).fit()
print(lm50.summary())


# In[59]:


print("predicted value()\n",lm50.fittedvalues)


# In[60]:


### APPLYING GLS to correct for heteroskedasticity


# In[61]:


#reminder of what out dataframe looks like at this point
#from code above
#regress price on lot, floor, bed
#expr1 = 'y ~ lot + floor + bed'
# get residual (uhat)
#uhat= y-yhat
#df['uhat_2'] = uhat * uhat
df20.shape


# In[62]:


#reminder of what out dataframe looks like at this point
df20.tail()


# In[63]:


#expr51 = 'df20.uhat_2 ~ df20.lotsize + df20.sqrft'
expr51 = 'uhat_2 ~ lotsize + sqrft'
lm51 = ols(expr51, df20).fit()
print(lm51.summary())


# In[64]:


#create the predicted value of y, note: uhat51 is not a residual
uhat51_fitted = lm51.predict()
#add to the dataframe
df20['uhat51_fitted'] = uhat51_fitted
#check the dataframe, previously was 13 columns, should have added 1 column for a total of 14
df20.shape
#df20.tail()


# In[65]:


#using numpy square root function
#add to the dataframe
df20['price_t'] = df20.price / np.sqrt(df20.uhat51_fitted)
df20['lotsize_t'] = df20.lotsize / np.sqrt(df20.uhat51_fitted)
df20['sqrft_t'] = df20.sqrft / np.sqrt(df20.uhat51_fitted)
df20['bdrms_t'] = df20.bdrms / np.sqrt(df20.uhat51_fitted)
df20['const_t'] = 1 / np.sqrt(df20.uhat51_fitted)
#check the dataframe, previously was 14 columns, should have added 5 columns for a total of 19
df20.shape
#df20.tail()


# In[66]:


#GLS
#transform by 1/sqrt(h)
#specify no constant with the -1 in the formula
#expr60 = 'df20.price_t ~ df20.lotsize_t + df20.sqrft_t + df20.bdrms_t + df20.const_t -1'
expr60 = 'price_t ~ lotsize_t + sqrft_t + bdrms_t + const_t -1'
lm60 = ols(expr60, df20).fit(hasconst=None)
print(lm60.summary())


# In[67]:


#create X5 data frame for next experiment
#from hprice1.dta
#setup 3 variables as a dataframe of independent variables for later use
exogdata5 = pd.read_stata(hprice1data,columns=['lotsize','sqrft','bdrms'],preserve_dtypes=True).values
X5 = pd.read_stata(hpv10data, preserve_dtypes=True)
X5 = pd.DataFrame(exogdata5)
X5.columns =['lotsize','sqrft','bdrms']
X5["constant"]=1
X5.shape
print(X5)
X5.columns.values
X5.head()
print(X5.dtypes)


# In[68]:


####  compare to statsmodels built in gls method  ###
### interesting!!!!, matches OLS with robust in lm40 ###
## should be the same as the one above and below ##
### GLS using stats models built in 
#gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)
#specify no constant in this regression
#gls_model1 = sm.GLS(df20.price, X5)
#gls_model1 = sm.GLS(df20.price, X5,hasconst=True)
gls_model1 = sm.GLS(df20.price, X5,hasconst=None)
gls_results1 = gls_model1.fit()
print(gls_results1.summary())


# In[ ]:





# In[ ]:





# In[69]:


### WLS (weight by 1/h)
#h(x)is some function of the explanatory variables in VAR(u|X)=sd*h(x)
#goal is to transform an equation with heteroskedastic errors to one with homoskedastic errors
#stata uses [aw=1/x] for analytic weights, statsmodels equivalent is supplying weignts in the .fit()
#first add the weight to the dataframe
#already done in cell ___df20['uhat51_fitted'] = df20.uhat51_fitted
wls_model = sm.WLS(df20.price, X5,weights=1./df20.uhat51_fitted)
wls_results = wls_model.fit()
print(wls_results.summary())


# In[70]:


#now we'll collect some of the above regressions and print in an excel table
#import the summary_col from the statsmodels library
from statsmodels.iolib.summary2 import summary_col
#example,dictionary of values to be called, .2f is a float with two decimals, d rounds to integer
info_dict1={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

results_table = summary_col(results=[lm20,lm30,lm40,lm50,lm51,lm60,gls_results1,wls_results],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['Model 20',
                                         'Model 30',
                                         'Model 40',
                                         'Model 50',
                                         'Model 51',
                                         'GLS Calc',
                                         'Built in GLS',
                                         'WLS Calc'],
                            info_dict=info_dict1,
                            regressor_order=['Intercept',
                                             'lotsize',
                                             'sqrft',
                                             'bdrms'])

results_table.add_title('Table 2 - GLS and WLS Regressions')

print(results_table)


# In[71]:


from pystout import pystout
#only statsmodels.ols().fit() will pass in
#so you have to exclude the GLS and WLS models
#ie lm60,gls_results1,wls_results, have been excluded
pystout(models=[lm20,lm30,lm40,lm50,lm51],
        file=r'C:\Users\jfras\OneDrive\DataScience\Python\table1a.csv',
        addnotes=['Selected models','using pystout'],
        digits=2,
        #endog_names=['lot', 'floor', 'bed', 'lot_floor', 'lot_bed', 'floor_bed', 'lot_2', 'floor_2', 'bed_2'],
        endog_names=True,
        #varlabels={'lot':'Lot size cat','floor':'sq feet','bed':'Num bedrooms', 
        #'lot_floor':'lotcat-sq feet','lot_bed':'lotcat-Num bedrooms','floor_bed':'sq feet-Num bedrooms', 
        #'lot_2':'lot cat sq','floor_2':'sq feet sq','bed_2':'Num bedrooms sq'},
        addrows={'Test':['A','Test','Row','Here','Too']},
        mgroups={'OLS':[1,3],'Prep':[4,5], 'GSL':[6,7],'WLS':8},
        #modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}','fvalue':'F-stat'}
        )
#this still put everthing in a cell
#https://pypi.org/project/pystout/


# In[72]:


from stargazer.stargazer import Stargazer
#from IPython.core.display import HTML
stargazer1 = Stargazer([lm20,lm30,lm40,lm50,lm51,lm60,gls_results1,wls_results])
stargazer1.significant_digits(2)
stargazer1.title("Table1s: Selected Regressions")
stargazer1
#stargazer.render_latex()
#stargazer.render_html()
#
#These lines were ignored
#stargazer1.type="text"
#stargazer1.out=r'C:\Users\jfras\OneDrive\DataScience\Python\table1s.csv'
#
#https://github.com/mwburke/stargazer
#https://web.northeastern.edu/econpress/wp-content/uploads/2016/04/Stargazer.pdf


# In[73]:


#now we'll collect some of the above regressions and print in an excel table
#import the summary_col from the statsmodels library
from statsmodels.iolib.summary2 import summary_col
#example,dictionary of values to be called, .2f is a float with two decimals, d rounds to integer
info_dict1={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

results_table = summary_col(results=[lm20,lm30,lm40,lm50,lm51,lm60,gls_results1,wls_results],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['Model 20',
                                         'Model 30',
                                         'Model 40',
                                         'Model 50',
                                         'Model 51',
                                         'Model 60 GLS',
                                         'GLS',
                                         'WLS',
                                        ],
                            info_dict=info_dict1,
                            regressor_order=['Intercept',
                                             'lot',
                                             'floor',
                                             'bed'])

results_table.add_title('Table 3 - OLS,GLS,WLS Regressions')

print(results_table)


# In[ ]:





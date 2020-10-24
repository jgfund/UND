#From econ 534 Heteroskedasticity lesson
#from Multiple Regression Section
#test for heteroskedasticity using the White (1980) test
#show how to use WLS/GLS to correct for hetereoskedasticity
#call in homepriceV10.dta data, .dta is a STATA file
#
#let's prepare a dataframe
# input Stata file
library(foreign)
housedata <- read.dta("C:/Users/jfras/OneDrive/UND/534AppliedEcon/Datasets/homepriceV10.dta")
df1 <- data.frame(housedata, stringsAsFactors=False)
#look at the dataframe
head (df1,10)
typeof(df1)
#check the data types
str(df1)
#look at some summary statistics
summary(df1)
library(pastecs)
stat.desc(df1)
#check dataframe shape
dim(df1)
#scatter plot of square feet and price
attach(df1)
plot(floor, price, main="Square Feet vs Price",
     xlab="Square Feet", ylab="Price", pch=19)
#can also use the car package
library(car)
scatterplot(price ~ floor | bed, data=df1, 
            xlab="Square Feet", ylab="Price", 
            main="Price vs Square Feet" 
            )
#took out "labels=row.names(housedata)" caused error
#ggplot2 scatterplot
#https://ggplot2.tidyverse.org/
#http://www.cookbook-r.com/Graphs/Plotting_distributions_(ggplot2)/
library(ggplot2)
# Basic scatter plot
#https://cran.rstudio.com/bin/windows/Rtools/
#then install RTools 
#install.packages("digest")
ggplot(df1, aes(x=floor, y=price)) + geom_point()
# Change the point size, and shape
ggplot(df1, aes(x=floor, y=price)) +
  geom_point(size=2, shape=23)
#check distribution
ggplot(df1, aes(x=price)) + 
  geom_histogram(binwidth=50,aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.15, fill="#FF6666") 
#setup 3 variables as a dataframe of independent variables for later use
X <- housedata[,c("floor","lot","bed")]
#add a constant to the dataframe
X['constant']=1
head(X,5)
#setup y as the dependent variable data frame for later use
pricedata <-df1['price']
y <- df1[,c("price")]
head(y,5)
typeof(y)
str(y)
#create the regression using stats models
# Multiple Linear Regression Example
lm1 <- lm(price ~ lot + floor + bed, data=df1)
summary(lm1) # show results
anova(lm1)
library(stargazer)
sum1sg <-stargazer(lm1, type="text", out="models.txt")
#print attributes of regression for later use, similair to STATA's "e"
#to list attributes available from regression
attributes(summary(lm1))
#results summaries of model fitting functions
#https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/summary
names(lm1)
summary(lm1)$r.squared
summary(lm1)$coefficients
summary(lm1)$residuals
#create variable of residuals, similair to STATA's "e"
df1residual <- resid(lm1)
#create variable of predicted values
df1predicted <- predict(lm1)
#add predicted value and residual value to the df1 dataframe
df1[["yhat"]] <-c(df1predicted)
df1[["uhat"]] <-c(df1residual)
head (df1,10)
#now plot the predicted price versus the actual price
#https://ggplot2.tidyverse.org/
ggplot(df1, aes(x=price, y=yhat)) +
  geom_point(size=2, shape=20) +
  labs(title="Actual Price vs Predicted Price" ,
       x="Price", y="Predicted Price") +
         theme_classic()
#plot the residuals
ggplot(df1, aes(x=yhat, y=uhat)) +
  geom_point(size=2, shape=21, fill="red") +
  labs(title="Actual Price vs Residual" ,
       x="Predicted Price", y="Residual") +
  theme_classic()
#start of long method for white test
#create some variables and add to dataframe
df1[["uhat_2"]] <- (df1$uhat * df1$uhat)
df1[['lot_floor']] <- (df1$lot * df1$floor)
df1[['lot_bed']] <- df1$lot * df1$bed
df1[['floor_bed']] <- df1$floor * df1$bed
df1[['lot_2']] <- df1$lot * df1$lot
df1[['floor_2']] <- df1$floor * df1$floor
df1[['bed_2']] <- df1$bed * df1$bed
dim(df1)
head(df1,10)
#using these interactive variables and the residual uhat from above
#regress residual squared on 'floor', 'lot', 'bed','lot_floor','lot_bed','floor_bed','lot_2','floor_2','bed_2'
lm2 <- lm(uhat_2 ~ lot + floor + bed + lot_floor + lot_bed + floor_bed + lot_2 + floor_2 + bed_2, data=df1)
summary(lm2) # show results
anova(lm2)
attributes(summary(lm2))
names(lm2)
#see what names are attached to the lm2 object
str(lm2)
#get attributes from lm2 object
gs <-summary(lm2)
gs$fstatistic
gs$fstatistic[2]
#extractor function
df.residual(lm2)
lm2$df.residual
lm2$coefficients
#example for extracting regression results into a dataframe
#tvalue is in 3rd position of anova table
summary(lm2)$coefficients[,3]

#https://jtools.jacob-long.com/reference/summ.lm.html
#https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html
summary(lm2)$r.squared
summary(lm2)$df
summary(lm2)$df[1]
summary(lm2)$df[2]
summary(lm2)$df[3]
#extracting an object from a list
lm2$df.residual
#lm2$residuals, to list all of the residuals

#assign r squared, degrees of freedom, and number of obs from lm2 model
rsquaredlm2 <- summary(lm2)$r.squared
degfreemodlm2 <- gs$fstatistic[2]
numobslm2 <- nobs(lm2)

#try to emulate STATA's di chi2tail(9, e(r2)*e(N)), #di is display
#--Description: the reverse cumulative (upper tail or survivor) 
#--X2 distribution with df degrees of
#--freedom; 1 if x < 0
#--chi2tail(df,x) = 1 - chi2(df,x)
#comparison for upcoming chi squared test
#find the critical value for a 95% confidence level
value <- qchisq(.95, degfreemodlm2)
print(paste("crifical value for 95% conficence level with 9 degrees of freedom ", value))
#confirm with cdf
library(PEIP)
p= chi2cdf(value, degfreemodlm2)
print(p)
#Chi2 test
arg2 <- rsquaredlm2*numobslm2
value2 <- chi2cdf(arg2,degfreemodlm2)
value3 <- (1-(chi2cdf(arg2,degfreemodlm2)))
print(value2)
print(value3)
#quick white tes
### on lm1 ###
#returns
#-Lagrange multiplier stat
#-p-value of Langrange multiplier test
#-F-Statistic of the hypothesis that error variance does not depend on x
#-p-value of the F statistic
#white test from het.test package
library(vars)
library(methods)
#H0: Homoskedasticity
#H1: Heteroskedasticity
#white test from htest package only for Vector Autoregressive models not lm's
#het.test::whites.htest(lm1)
#white test from skedastic package
library(skedastic,interaction(TRUE))
white_lm(lm1, interactions = TRUE)
#try use breuschpagan test on lm2 to do white test from the lmtest package
library(lmtest)
bptest(lm2, data=df1)
#quick breuschpagan test bptest() from the lmtest package
bptest(lm1, data=df1)
#robust standard errors
#regress price on lot, floor, bed
library(estimatr)
lm3 <- lm_robust(price ~ lot + floor + bed, data=df1,se_type = 'HC1')
#lm3 is not an lm object, eststo only support lm objects
summary(lm3)
#now we'll collect the two above regressions and print in an excel table
library(estout)
estclear()
eststo(lm1)
eststo(lm2)
esttab(filename="results_table1r",csv=TRUE,caption="Table 1")
#print(esttab)
#can also use this
library(jtools)
library(huxtable)
export_summs(lm1, lm2, scale = FALSE)
#heteroskadasticy robust standard errors
#requires sandwich package
library(sandwich)
coeftest(lm1,vcov=vcovHC(lm1,type="HC1"))
### HERE WE START USING A NEW DATA file, has log of some fields ###
hprice1data <- read.dta("C:/Users/jfras/OneDrive/UND/534AppliedEcon/Datasets/hprice1.dta")
housedata2 <- data.frame(hprice1data, stringsAsFactors=False)
df20 <- data.frame(housedata2, stringsAsFactors=False)
#add a constant for later use
df20[['constant']]=1
#look at the dataframe
head (df20,10)
typeof(df20)
#check the data types
str(df20)
#look at some summary statistics
summary(df20)
library(pastecs)
stat.desc(df20)
#check dataframe shape
dim(df20)
lm20 <- lm(price ~ lotsize + sqrft + bdrms, data=df20)
summary(lm20) # show results
anova(lm20)
#get attributes from lm20 object
attributes(summary(lm20))
#see what names are attached to the lm20 object
names(lm20)
#illustrates syntax for later use of attributes
gs20 <-summary(lm20)
gs20$fstatistic
gs20$fstatistic[2]
#extractor function
df.residual(lm20)
lm20$df.residual
lm20$coefficients
#example for extracting regression results into a dataframe
#tvalue is in 3rd position of anova table
summary(lm20)$coefficients[,3]
#predicted value
#create variable of predicted values
df20predicted <- predict(lm20)
df20residual <- summary(lm20)$residuals
#add predicted value and residual value to the df20 dataframe
df20[["yhat"]] <-c(df20predicted)
df20[["uhat"]] <-c(df20residual)
head (df20,10)
#from hprice1.dta
#setup 3 variables as a dataframe of independent variables for later use
exogdata21 <- hprice1data[,c('lotsize','sqrft','bdrms')]
X21 <- data.frame(exogdata21, stringsAsFactors=FALSE)
X21['constant']=1
dim.data.frame(X21)
head(X21,5)
typeof(X21)
str(X21)
#white test on prior regression before running new one
white_lm(lm20, interactions = TRUE)
#now do a regression using logs
lm30 <- lm(lprice ~ llotsize + lsqrft + bdrms, data=df20)
summary(lm30) # show results
anova(lm30)
typeof(lm30) #produces a list object
class(lm30) #produces an "lm" object
#from hprice1.dta
#setup 3 variables as a dataframe of independent variables for later use
exogdata31 <- hprice1data[,c('llotsize','lsqrft','bdrms')]
X31 <- data.frame(exogdata31,stringsAsfactors=FALSE)
colnames(X31) <- c("log lotsize","log sqrft","bdrms")
X31['constant']=1
dim.data.frame(X31)
head(X31,5)
typeof(X31)
str(X31)
#white test
white_lm(lm30, interactions = TRUE)
#now do a regression using robust standard errors on df20 data from hprice1
#regression using robust standard errors
#https://data.princeton.edu/wws509/r/robust
#https://declaredesign.org/r/estimatr/reference/lm_robust.html
#NOTE: lm_robust output object is not the same object as an lm
library(estimatr)
lm40 <- lm_robust(price ~ lotsize + sqrft + bdrms, data=df20,se_type = 'HC1')
summary(lm40)
glance(lm40)
typeof(lm40)
class(lm40) #produces lm_robust object
#NOTE: the lm_robust is incompatible with eststo, esttab
#can use texreg for latex ouput https://ditraglia.com/econ224/lab07.pdf
#NOTE: tidy(lm40) #creates a dataframe from an lm_robust object
#https://cran.rapporter.net/web/packages/estimatr/estimatr.pdf
#try mass package
library(foreign)
library(MASS)
lm41 <- rlm(price ~ lotsize + sqrft + bdrms, data=df20,se_type = 'HC1')
summary(lm41)
class(lm41) #produces a "rlm" "lm" object
#try robustbase package
library(robustbase)
lm42 <- lmrob(price ~ lotsize + sqrft + bdrms, data=df20,se_type = 'HC1')
summary(lm42)
class(lm42) #produces a lmrob object
#going back to the df20 dataframe (from the hprice1.dta dataset)
#add uhat_2 to dataframe 20
df20[["uhat_2"]] <- (df20$uhat * df20$uhat)
head(df20)
#regress residual squared on jus lotsize and sqrtft
lm50 <- lm(uhat_2 ~ lotsize + sqrft, data=df20)
summary(lm50) # show results
anova(lm50)
#generate predicted value from lm50
predict.lm(lm50)
### APPLYING GLS to correct for heteroskedasticity
tail(df20,5)
lm51 <- lm(uhat_2 ~ lotsize + sqrft, data=df20)
summary(lm51)
#create the predicted value of y, note: uhat51 is not a residual
#and put in dataframe20
df20[["uhat51_fitted"]] <- (predict.lm(lm51))
head(df20,5)
#add to the dataframe
df20[["price_t"]] <- (df20$price/sqrt(df20$uhat51_fitted))
df20[["lotsize_t"]] <- (df20$lotsize/sqrt(df20$uhat51_fitted))
df20[["sqrft_t"]] <- (df20$sqrft/sqrt(df20$uhat51_fitted))
df20[["bdrms_t"]] <- (df20$bdrms/sqrt(df20$uhat51_fitted))
df20[["const_t"]] <- (1/sqrt(df20$uhat51_fitted))
head(df20,5)
dim.data.frame(df20)
#GLS
#transform by 1/sqrt(h)
#specify no constant with the -1 in the formula
lm60 <- lm(df20$price_t ~ df20$lotsize_t + df20$sqrft_t + df20$bdrms_t + df20$const_t -1)
summary(lm60)
#create X5 data frame for next experiment
#from hprice1.dta
#setup 3 variables as a dataframe of independent variables for later use
exogdata5 <- hprice1data[,c('lotsize','sqrft','bdrms')]
X5 <- data.frame(exogdata5)
colnames(X5) <- c("lotsize","sqrft","bdrms")
X5['constant']=1
dim.data.frame(X5)
head(X5,5)
typeof(X5)
str(X5)
#create a dataframe for price
X5p <- data.frame(hprice1data$price)
colnames(X5p) <- c("price")
head(X5p,5)
str(X5p)
####  compare to statsmodels built in gls method  ###
### interesting!!!!, matches OLS with robust in lm40 ###
## should be the same as the one above and below ##
### GLS using nlme package 
#specify no constant in this regression
#https://www.rdocumentation.org/packages/nlme/versions/3.1-149
library(nlme)
gls_model1 <- gls(price ~ lotsize + sqrft + bdrms + constant -1, data=df20)
summary(gls_model1)
### WLS (weight by 1/h)
#h(x)is some function of the explanatory variables in VAR(u|X)=sd*h(x)
#goal is to transform an equation with heteroskedastic errors to one with homoskedastic errors
#stata uses [aw=1/x] for analytic weights, statsmodels equivalent is supplying weignts in the .fit()
#R uses 
#https://www.rdocumentation.org/packages/UStatBookABSC/versions/1.0.0/topics/WLS
#first add the weight to the dataframe
#already done in cell ___df20['uhat51_fitted'] = df20.uhat51_fitted
head(df20,5)
#WLS model
#https://rpubs.com/mpfoley73/500818
#weights1 <- 1./sqrt(df20$uhat51_fitted) #not used gives wrong result
weights2 <- 1./(df20$uhat51_fitted)
wls_model <- lm(price ~ lotsize + sqrft + bdrms,weights = weights2, data=df20)
summary(wls_model)
#another method for wls (not used)
#https://www.rdocumentation.org/packages/metaSEM/versions/1.2.4
#now we'll collect some of the above regressions and print in an excel table
#https://www.rdocumentation.org/packages/estout/versions/0.7-1/topics/esttab
#make sure we have a value for each model for error checking
is.na(lm40)
library(estout)
estclear()
eststo(lm20)
eststo(lm30)
#eststo(lm40) #lm_robust() incompatible with esttab
eststo(lm41)
eststo(lm50)
eststo(lm51)
eststo(lm60)
#eststo(gls_model1) #gls() incompatible with esttab
eststo(wls_model) #used lm with weights
esttab(filename="results_table1r2",csv=TRUE, colnumber = TRUE, caption="Table 2") #appears to only work with lm, not lm_robust or gls
#can also use this
library(jtools)
library(huxtable)
#export_summs does not appear to work for gls or wls models
export_summs(lm20,lm30,lm40,lm50,lm51,lm60, 
             model.names = c("Model 20", "Model 30", "Model 40", 
                             "Model 50", "Model 51", "Model 60"), 
             scale = FALSE)

#https://ditraglia.com/econ224/lab07.pd
#tried this for the objects produced that are not lm objects
#produces latex results
library(texreg)
texreg(list(lm40, gls_model1,wls_model), include.ci = FALSE,
       caption = 'Predicting House Prices',
       #custom.coef.names = c('not used 1','not used 2','not used 3')
       custom.note = 'Robust Standard Errors')

#try stargazer, #won't work for robust_lm objects
#library(stargazer)
#stargazer1 = stargazer(lm40, csv=TRUE)
#stargazer1 = Stargazer([lm20,lm30,lm40,lm50,lm51,lm60,gls_results1,wls_results])
#stargazer1.significant_digits(2)
#stargazer1.title("Table1s: Selected Regressions")
#stargazer1


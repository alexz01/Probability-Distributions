# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:13:54 2017

@author: aumale
"""

import openpyxl as opxl
import numpy as npy
import scipy.stats as spys
import matplotlib.pyplot as plt

UBitName = 'aumale'
personNumber = 50429040

print('UBitName = ',UBitName)
print('personNumber = ', personNumber,'\n')
dataFileLoc = './UniversityData.xlsx'

try:
    xlbook = opxl.load_workbook(dataFileLoc)
    xlsheets = xlbook.get_sheet_names()

except FileNotFoundError as fnfError:
    print(fnfError)

#get acctive excel sheet
xlsheet = xlbook.active

heading = list(xlsheet.rows)[0]

for index in range(len(heading)):
    if heading[index].value.find("CS Score")>=0 :
        CSScore_colm = index
    elif heading[index].value.find("Research Overhead")>=0 :
        ResearchOverhead_colm = index
    elif heading[index].value.find("Admin Base Pay")>=0 :
        AdminBasePay_colm = index
    elif heading[index].value.find("Tuition")>=0 :
        Tuition_colm = index

CSScore = []
ResearchOverhead = []
AdminBasePay = []
Tuition = []

for score in list(xlsheet.columns)[CSScore_colm]\
        [1:len(list(xlsheet.columns)[CSScore_colm])-1]:
    CSScore.append(float(score.value))
for overhead in list(xlsheet.columns)[ResearchOverhead_colm]\
        [1:len(list(xlsheet.columns)[ResearchOverhead_colm])-1]:
    ResearchOverhead.append(float(overhead.value))
for basepay in list(xlsheet.columns)[AdminBasePay_colm]\
        [1:len(list(xlsheet.columns)[AdminBasePay_colm])-1]:
    AdminBasePay.append(float(basepay.value))
for tuition in list(xlsheet.columns)[Tuition_colm]\
        [1:len(list(xlsheet.columns)[Tuition_colm])-1]:
    Tuition.append(float(tuition.value))

#Vector X containing the data variables
X = npy.array([CSScore,ResearchOverhead,AdminBasePay,Tuition])

#####################################################################
mu1 = npy.mean(X[0])
mu2 = npy.mean(X[1])
mu3 = npy.mean(X[2])
mu4 = npy.mean(X[3])

print('mu1 = ','{:.3f}'.format(mu1))
print('mu2 = ','{:.3f}'.format(mu2))
print('mu3 = ','{:.3f}'.format(mu3))
print('mu4 = ','{:.3f}'.format(mu4),'\n')

mu = npy.asarray([mu1,mu2,mu3,mu4])

var1 = npy.var(X[0])
var2 = npy.var(X[1])
var3 = npy.var(X[2])
var4 = npy.var(X[3])

print('var1 = ','{:.3f}'.format(var1))
print('var2 = ','{:.3f}'.format(var2))
print('var3 = ','{:.3f}'.format(var3))
print('var4 = ','{:.3f}'.format(var4),'\n')

var = npy.asarray([var1,var2,var3,var4])

sigma1 = npy.std(X[0])
sigma2 = npy.std(X[1])
sigma3 = npy.std(X[2])
sigma4 = npy.std(X[3])

print('sigma1 = ','{:.3f}'.format(sigma1))
print('sigma2 = ','{:.3f}'.format(sigma2))
print('sigma3 = ','{:.3f}'.format(sigma3))
print('sigma4 = ','{:.3f}'.format(sigma4),'\n')

sigma = npy.asarray([sigma1,sigma2,sigma3,sigma4])

#calculate covariance of normalized matrix
covarianceMat = npy.cov(X).round(3)

#calculate corelation cofficient matrix
correlationMat = npy.corrcoef(X).round(3)

#set format to print upto three digits of floating numbers
npy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print('covarianceMat = \n',covarianceMat,'\n')
# reset print options
npy.set_printoptions()
print('correlationMat =\n',correlationMat,'\n')

# calculating the univariate probability density function for each variable in X
PX = npy.exp(-0.5*(((X.T-mu)**2)/var))/(npy.sqrt(2*npy.pi*var))

#calclating the log likelyhood
logLikelihood = npy.sum(npy.log(PX))

print('logLikelihood = ','{:.3f}'.format(logLikelihood),'\n')

#calculating the multivariate probability density function
#multv_likelyhood = spys.multivariate_normal.pdf(X.T,mu,covarianceMat)

#when data is not normalized with mu and sigma
multv_likelyhood = spys.multivariate_normal.pdf(X.T,mu,covarianceMat,allow_singular=True)

#calculating the log likelihood considering data as multivariate
multivLogLikelihood = npy.sum(npy.log(multv_likelyhood))

print('multivariateLogLikelihood = ', '{:.3f}'.format(multivLogLikelihood))
#########################################################################################
#plotting data

#normalize the data
Xn =((X.T - X.T.min(axis=0))/(X.T.max(axis=0)-X.T.min(axis=0))).T.round(3)

mu_n = npy.asarray([npy.mean(Xn[0]),npy.mean(Xn[1]),npy.mean(Xn[2]),npy.mean(Xn[3])])
var_n = npy.asarray([npy.var(Xn[0]),npy.var(Xn[1]),npy.var(Xn[2]),npy.var(Xn[3])])
sigma_n = npy.asarray([npy.std(X[0]),npy.std(X[1]),npy.std(X[2]),npy.std(X[3])])
PXn = npy.exp(-0.5*(((Xn.T-mu_n)**2)/var_n))/(npy.sqrt(2*npy.pi*var_n))

#plot graph between independent variables and their probability density
plt.figure(figsize=(8,8))
plt.xlabel('Data points')
plt.ylabel('Probability')
CSScore_plot , = plt.plot(Xn[0],PXn.T[0],'bo')
ResearchOverhead_plot, = plt.plot(Xn[1],PXn.T[1],'b^')
AdminBasePay_plot, = plt.plot(Xn[2],PXn.T[2],'bs')
Tuition_plot, = plt.plot(Xn[3],PXn.T[3],'b.')

label_vector = ['CS Score', 'Research Overhead', 'Admin Base Pay','Tuition' ]
plt.legend([CSScore_plot,ResearchOverhead_plot,AdminBasePay_plot,Tuition_plot],
           label_vector)

#plot scatter graph for pairwise comparison
f,fig_arr = plt.subplots(4,4,figsize=(20,20))
for i, Xi in enumerate(Xn):
    for j, Xj in enumerate(Xn):
        if Xi is not Xj :
            fig_arr[i,j].scatter(Xi,Xj)
            fig_arr[i,j].set_xlabel(label_vector[i])
            fig_arr[i,j].set_ylabel(label_vector[j])

#plot heatmap of correlation for visual comparison
plt.matshow(correlationMat,cmap='gray')
plt.xlabel('$\sigma_i,_j$',fontsize=16)
plt.ylabel('$\sigma_j,_i$',fontsize=16)
plt.colorbar()

plt.figure()
plt.plot(multv_likelyhood,'k-')
plt.xlabel('data')
plt.ylabel('multivariate pdf')

plt.matshow(npy.cov(Xn),cmap='gray')
plt.colorbar()

plt.show()
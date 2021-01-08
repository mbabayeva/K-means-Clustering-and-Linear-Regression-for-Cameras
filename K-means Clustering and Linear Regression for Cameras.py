#!/usr/bin/env python
# coding: utf-8

#                                         1
#     

# In[3]:


import pandas as pd
import matplotlib as plt
import numpy as np
data = pd.read_csv('Camera.csv',sep=';')
df=pd.DataFrame(data)
df


# In[4]:


a=df.drop(df.index[0])
#a = a.drop('Release date', axis=1)
a = a.drop('Low resolution', axis=1)
a = a.drop('Storage included', axis=1)
#StorageIncluded variable contains individuals of different units(in range of 128-450) such as megabytes and gigabytes. 
a


# In[5]:


a.isnull().any()


# In[6]:


a[a.isnull().any(axis=1)]


# In[7]:


a.drop(a.index[345:347], inplace=True)


# In[8]:


a[['Release date','Max resolution','Effective pixels','Zoom wide (W)','Zoom tele (T)','Normal focus range',
   'Macro focus range','Weight (inc. batteries)','Dimensions','Price']] = a[['Release date',
    'Max resolution', 'Effective pixels','Zoom wide (W)','Zoom tele (T)','Normal focus range','Macro focus range',
    'Weight (inc. batteries)','Dimensions','Price']].astype(float)


# In[9]:


a=a[a['Weight (inc. batteries)'] != 0]
a=a[a['Max resolution'] != 0]
a=a[a['Dimensions'] != 0]
a=a[a['Normal focus range']!=0]
a=a[a['Macro focus range']!=0]
a=a[a['Effective pixels']!=0]


# In[10]:


a


# In[11]:


a.set_index('Model', inplace=True)


# In[12]:


#Frequency Histograms of each Variables


# In[13]:


#Histograms to show the distribution of the variable values


# In[14]:


a.describe()


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Max resolution'].hist(bins=10);


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Release date'].hist(bins=10);


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Effective pixels'].hist(bins=10);


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Zoom wide (W)'].hist(bins=10);


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Zoom tele (T)'].hist(bins=10);


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Normal focus range'].hist(bins=10);


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Macro focus range'].hist(bins=10);


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Weight (inc. batteries)'].hist(bins=10);


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Dimensions'].hist(bins=10);


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
a['Price'].hist(bins=10);


# In[25]:


#The data should be scaled  in further analysis, because of different range of values


# In[26]:


#Drawing boxplot for each Variables


# In[27]:


#Boxplots for each of the variables as another indicator of spread.


# In[28]:


a.describe()


# In[29]:


a.boxplot(column=['Max resolution']);


# In[30]:


a.boxplot(column=['Release date']);


# In[31]:


a.boxplot(column=['Effective pixels']);


# In[32]:


a.boxplot(column=['Zoom wide (W)']);


# In[33]:


a.boxplot(column=['Zoom tele (T)']);


# In[34]:


a.boxplot(column=['Normal focus range']);


# In[35]:


a.boxplot(column=['Macro focus range']);


# In[36]:


a.boxplot(column=['Weight (inc. batteries)']);


# In[37]:


a.boxplot(column=['Dimensions']);


# In[38]:


a.boxplot(column=['Price']);


# In[ ]:





# In[39]:


#Creating  the Correlation Matrix


# In[40]:


Correlations = a.corr()
Correlations


# In[41]:


#Representing Correlation matrix with __matplotlib__


# In[42]:


import matplotlib.pyplot as plt
import numpy as np

names = list(Correlations.columns)
fig = plt.figure(figsize=[40,20])
ax = fig.add_subplot(111)
cax = ax.matshow(Correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,10,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[43]:


#The three most related pairs of variables are determined from the graph and according to graph properties(dark yellow&purple color corresponds to more correlation between pairs)I have choosen the following pairs :
#Max resolution and Effective pixels
#Low Resolution andMax Resoluion
#Weight(inc.batteries) and Zoom wide
#in other words these 3 combinations are correlated positively, their scatter plot shows increasing relation 


# In[44]:


import matplotlib.pyplot as plt

plt.scatter(Correlations['Max resolution'], Correlations['Effective pixels'])


# In[45]:


import matplotlib.pyplot as plt

plt.scatter(Correlations['Effective pixels'], Correlations['Release date'])


# In[46]:


import matplotlib.pyplot as plt

plt.scatter(Correlations['Release date'], Correlations['Max resolution'])


# In[ ]:





#                                             2

# In[ ]:





# In[47]:


# Computation of Sample Mean and Sample Standard Deviation of 'Max resolution'

import numpy as np
import random
bW=np.random.choice(a['Max resolution'].values,10) #random sample(of length 10) of this variable is made
print(bW)
print('Sample Mean',np.mean(bW))
print('Sample Standard Deviation',np.std(bW))


# In[ ]:





# In[48]:


# Computation of Sample Mean and Sample Standard Deviation of 'Effective pixels'
import numpy as np
import random
b=np.random.choice(a['Effective pixels'].values,10)
print(b)
print('Sample Mean',np.mean(b))
print('Sample Standard Deviation',np.std(b))


# In[49]:


# Computation of Sample Mean and Sample Standard Deviation of 'Zoom wide'
import numpy as np
import random
DR=np.random.choice(a['Zoom wide (W)'].values,10)
print(DR)
print('Sample Mean',np.mean(DR))
print('Sample Standard Deviation',np.std(DR))


# In[50]:


# Computation of Sample Mean and Sample Standard Deviation of 'Zoom tele'
import numpy as np
import random
DR=np.random.choice(a['Zoom tele (T)'].values,10)
print(DR)
print('Sample Mean',np.mean(DR))
print('Sample Standard Deviation',np.std(DR))


# In[51]:


# Computation of Sample Mean and Sample Standard Deviation of 'Normal focus range'
import numpy as np
import random
DR=np.random.choice(a['Normal focus range'].values,10)
print(DR)
print('Sample Mean',np.mean(DR))
print('Sample Standard Deviation',np.std(DR))


# In[52]:


# Computation of Sample Mean and Sample Standard Deviation of 'Macro focus range'
import numpy as np
import random
DR=np.random.choice(a['Macro focus range'].values,10)
print(DR)
print('Sample Mean',np.mean(DR))
print('Sample Standard Deviation',np.std(DR))


# In[53]:


# Computation of Sample Mean and Sample Standard Deviation of 'Weight (inc.batteries)'
import numpy as np
import random
DR=np.random.choice(a['Weight (inc. batteries)'].values,10)
print(DR)
print('Sample Mean',np.mean(DR))
print('Sample Standard Deviation',np.std(DR))


# In[54]:


# Computation of Sample Mean and Sample Standard Deviation of 'Dimensions'
import numpy as np
import random
DR=np.random.choice(a['Dimensions'].values,10)
print(DR)
print('Sample Mean',np.mean(DR))
print('Sample Standard Deviation',np.std(DR))


# In[55]:


# Computation of Sample Mean and Sample Standard Deviation of 'Price'
import numpy as np
import random
DR=np.random.choice(a['Price'].values,10)
print(DR)
print('Sample Mean',np.mean(DR))
print('Sample Standard Deviation',np.std(DR))


#                                       3

# In[56]:


#python fuction  that  computes confidence  interval for  the  population  mean under the assumption that all are normally distributed

 


# In[57]:


import numpy as np
def ConfidenceIntervalForMean(data,percentage):
    m=np.mean(data)
    standardError=(np.std(data))/((len(data))**(1/2))
    
    if percentage==95:  #mu+2sigma
        h=1.96*standardError
    elif percentage==68: #mu+1sigma
        h=0.9945*standardError
    elif percentage==99: #mu+3sigma
        h=2.575*standardError
    return m-h,m+h
print(ConfidenceIntervalForMean([5,7,14,22,35],68))
print(ConfidenceIntervalForMean([5,7,14,22,35],95))
print(ConfidenceIntervalForMean([5,7,14,22,35],99))


#                                          4

# In[58]:


#Finding confidence interval for each variable
#the computation of confidence interval can be found also in R file called #4Finding Confidence Interval for each Variable


# In[59]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Max resolution
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Max resolution'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Max resolution'].values,99))


# In[60]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Effective pixels
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Effective pixels'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Effective pixels'].values,99))


# In[61]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Zoom wide
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Zoom wide (W)'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Zoom wide (W)'].values,99))


# In[62]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Zoom tele
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Zoom tele (T)'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Zoom tele (T)'].values,99))


# In[63]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Normal focus range
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Normal focus range'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Normal focus range'].values,99))


# In[64]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Macro focus range
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Macro focus range'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Macro focus range'].values,99))


# In[65]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Weight (inc.batteries)
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Weight (inc. batteries)'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Weight (inc. batteries)'].values,99))


# In[66]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Dimensions
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Dimensions'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Dimensions'].values,99))


# In[67]:


#When alpha is 0.05 then z-score is 1.96 >>95%
#When alpha is 0.01 then z-score is 2.575 >>99%
#100(1-alpha)
                                 #Price
#alpha=0.05
print('alpha=0.05>>ConfidenceInterval',ConfidenceIntervalForMean(a['Price'].values,95))
#alpha=0.01
print('alpha=0.01>>ConfidenceInterval',ConfidenceIntervalForMean(a['Price'].values,99))


# In[68]:


#All above computations are made also via R.
#-#4-Finding Confidence Interval for each Variable - R file


#                                              5

# In[69]:


#Computing Empirical Confidence Interval


# In[70]:


#Checking the variable with Largest Variation Coefficient
 
M=[a['Max resolution'].values.std()/a['Max resolution'].values.mean(),a['Release date'].values.std()/a['Release date'].values.mean(),a['Effective pixels'].values.std()/a['Effective pixels'].values.mean(),a['Zoom wide (W)'].values.std()/a['Zoom wide (W)'].values.mean(),a['Zoom tele (T)'].values.std()/a['Zoom tele (T)'].values.mean(),a['Normal focus range'].values.std()/a['Normal focus range'].values.mean(),a['Macro focus range'].values.std()/a['Macro focus range'].values.mean(),a['Weight (inc. batteries)'].values.std()/a['Weight (inc. batteries)'].values.mean(),a['Dimensions'].values.std()/a['Dimensions'].values.mean(),a['Price'].values.std()/a['Price'].values.mean()]
for i in range(len(M)):
    print('Largest Variation Coefficient is  ', np.max(M), ' >> ', M[i] )

#Output: Price variable has the Largest Variation Coefficient


# In[71]:


#Empirical Confidence Interval for Population Mean


# In[104]:


sample_data=a['Price'].values
n = len(sample_data)
repetitions = 10000
resampled_data = np.random.choice(sample_data, (n, repetitions))
MeansOfResampledData = resampled_data.mean(axis=0)
MeansOfResampledData.sort()

print('Empirical Confidence Interval for Population Mean,alpha=0.01>>99%, is',np.percentile(MeansOfResampledData, [0.5, 99.5]))


# In[73]:


#Empirical Confidence Interval for Population Median


# In[74]:


sample_data=a['Price'].values
n = len(sample_data)
repetitions = 10000
resampled_data = np.random.choice(sample_data, (n, repetitions))
MediansOfResampledData = np.median(resampled_data,axis=0)
MediansOfResampledData.sort()

print('Empirical Confidence Interval for Population Median, alpha=0.01>>>99%, is',np.percentile(MediansOfResampledData, [0.5, 99.5]))


# In[ ]:





#                                           6 

# In[ ]:





# In[75]:



#Testing the null hypothesis that the population mean is equal to sample median - alpha=0.01 


# In[76]:


Mydata=a['Price'].values
sample_data=np.random.choice(Mydata,100)
n = len(Mydata)
repetitions = 10000
resampled_data = np.random.choice(Mydata, (n, repetitions))
MediansOfResampledData.sort()
#z score = +/-2.58 >>> when alpha=0.01
#Computing test statistics z=(xBar-mean)/(standardDeviation/sqrt(sample_size))
MyResampledData_Mean=np.median(resampled_data) #<<<the population mean is equal to sample median 
z=(sample_data.mean()-MyResampledData_Mean)/((resampled_data.std())/((len(sample_data))**(1/2)))
if (z<-2.58)|(z>2.58):
    print('Reject the Null Hypothesis')
else:
    print('Accept the Null Hypothesis')


# In[ ]:





#                                            7

# In[77]:


#7 can be found in R file and in PDF


#                                            
#                                            8
#                                           

# #8 tested using R  and python. The only difference is that the K-means algorithm is applied on data itself in R file(
# description in PDF). However, in this python code Principal Components(dimension reduction) was considered.

# In[78]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
data=a;
data_clean = data.dropna()
cluster=data_clean[['Max resolution', 'Effective pixels', 'Zoom wide (W)',
       'Zoom tele (T)', 'Normal focus range', 'Macro focus range', 'Weight (inc. batteries)',
       'Dimensions']]
cluster.describe()


#  A series of k-means cluster analyses were conducted on the training data specifying k=2-20 clusters, using Euclidean distance. The variance in the clustering variables that was accounted for by the clusters (r-square) was plotted for each of the 20 cluster solutions in an elbow curve to provide guidance for choosing the number of clusters to interpret.

# In[79]:



#standardizing clustering variables to have mean=0 and standard deviation=1
clustervar=cluster.copy()
clustervar['Max resolution']=preprocessing.scale(clustervar['Max resolution'].astype('float64'))
clustervar['Effective pixels']=preprocessing.scale(clustervar['Effective pixels'].astype('float64'))
clustervar['Zoom wide (W)']=preprocessing.scale(clustervar['Zoom wide (W)'].astype('float64'))
clustervar['Zoom tele (T)']=preprocessing.scale(clustervar['Zoom tele (T)'].astype('float64'))
clustervar['Normal focus range']=preprocessing.scale(clustervar['Normal focus range'].astype('float64'))
clustervar['Macro focus range']=preprocessing.scale(clustervar['Macro focus range'].astype('float64'))
clustervar['Weight (inc. batteries)']=preprocessing.scale(clustervar['Weight (inc. batteries)'].astype('float64'))
clustervar['Dimensions']=preprocessing.scale(clustervar['Dimensions'].astype('float64'))


#splitting data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)


# In[80]:


# k-means cluster analysis for 2-20 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(2,20)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
                    / clus_train.shape[0])


# In[81]:


#Plotting average distance from observations from the cluster centroid
#I used the Elbow Method to identify number of clusters to choose
import matplotlib.pyplot as plt
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.plot(clusters, meandist)
plt.show()


# In[82]:


#5 clusters solution was interpreted
m=KMeans(n_clusters=5)
m.fit(clus_train)
clusassign=m.predict(clus_train)


# In[ ]:





# In[83]:


# plotting clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=m.labels_,)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


# In[ ]:





#                                 Implementing Simple Linear Regression

# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[85]:


m=a[['Max resolution', 'Price']]
m.plot(kind='scatter', x='Max resolution', y='Price', figsize=(12,8))


# In[86]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
c=a.values
# Use only one feature
a_X=c[:, np.newaxis, 1] #Max resolution

# Split the data into training/testing sets
a_X_train = a_X[:-20]
a_X_test = a_X[-20:]
#a_Y=np.transpose(np.matrix(a.Price))
a_Y=c[:,9]
# Split the targets into training/testing sets
a_y_train = a_Y[:-20]
a_y_test = a_Y[-20:]
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(a_X_train, a_y_train)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(a_X_test) - a_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(a_X_test, a_y_test))
# Plot outputs

plt.scatter(x=a_X_test, y=a_y_test,  color='black')

plt.plot(a_X_test, regr.predict(a_X_test), color='blue',linewidth=3)
plt.xlabel('TrueValues')
plt.ylabel('Predictions')
plt.xticks(())
plt.yticks(())

plt.show()


# In[88]:


a.columns=['Releasedate','MaxResolution','EffectivePixels','ZoomWide','ZoomTelescope','NormalFocusRange','MacroFocusRange','Weight','Length','Price']


# In[131]:


# Split the data into training/testing sets randomly
from sklearn.utils import shuffle
a = shuffle(a)
a_train=a[:-80]
a_test=a[-80:]


# In[89]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# visualizing the relationship between the features(variables) and the response(target) using scatterplots
fig, axs = plt.subplots(1, 3, sharey=True)
a.plot(kind='scatter', x='Weight', y='Price', ax=axs[0], figsize=(25, 9))
a.plot(kind='scatter', x='Releasedate', y='Price', ax=axs[1])
a.plot(kind='scatter', x='ZoomWide', y='Price', ax=axs[2])


#                                           Simple Linear Regression

# In[90]:


import statsmodels.api as sm


# In[188]:


import statsmodels.formula.api as smf


# In[187]:


# Create a fitted model
plt.scatter(a.Weight, a.Price, alpha=0.3)
plt.xlabel('Weight')
plt.ylabel('Price')

Weight_linspace=np.linspace(a.Weight.min(),a.Weight.max(),80)

# create a fitted model in one line
lm = smf.ols(formula='Price ~ Weight',data=a_train).fit()

plt.plot(Weight_linspace, lm.params[0] + lm.params[1] * Weight_linspace, 'r')
plt.plot(a_test.Weight, lm.predict(a_test.Weight), 'g')
print(lm.summary())
pd.concat([a_test.Price, lm.predict(a_test.Weight)], axis=1)


# In[189]:


# Create a fitted model
plt.scatter(a.MaxResolution, a.Price, alpha=0.3)
plt.xlabel('ZoomWide')
plt.ylabel('Price')

ZoomWide_linspace=np.linspace(a.ZoomWide.min(),a.ZoomWide.max(),80)

# create a fitted model in one line
lm = smf.ols(formula='Price ~ ZoomWide',data=a_train).fit()

plt.plot(ZoomWide_linspace, lm.params[0] + lm.params[1] * ZoomWide_linspace, 'r')
plt.plot(a_test.ZoomWide, lm.predict(a_test.ZoomWide), 'g')
print(lm.summary())
pd.concat([a_test.Price, lm.predict(a_test.ZoomWide)], axis=1)


#                             Multiple Linear Regression 
#                       
#     

# In[97]:


# create a fitted model 9 features
# generating full model
lm = smf.ols(formula='Price ~ Weight + Releasedate + ZoomWide + Length + MaxResolution + EffectivePixels + ZoomTelescope + NormalFocusRange + MacroFocusRange', data=a).fit()

# print the coefficients
print(lm.summary())


# We could trim some variables from the model using the p-value. Using backward elimination with a p-value cutoff 0.05 (start with the full model and trim the predictors with p-values greater than 0.05), we ultimately eliminate the Releasedate, Length, MaxResolution, EffectivePixels, NormalFocusRange and MacrofocusRange predictors. We have a smaller model as follows:  

# In[98]:


# constructing a smaller model
lm = smf.ols(formula='Price ~ Weight + ZoomWide + ZoomTelescope', data=a).fit()

# print the coefficients

print(lm.summary())


# In[99]:


# constructing a smaller model
lm = smf.ols(formula='Price ~ Weight + NormalFocusRange + ZoomTelescope', data=a).fit()

# print the coefficients

print(lm.summary())


# In[ ]:





# In[100]:



from mpl_toolkits.mplot3d import Axes3D

X = a[['Weight', 'ZoomWide']]
y = a['Price']

## fit a OLS model with intercept on Weight and ZoomWide
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

## Create the 3d plot 
# Weight/Zoomwide grid for 3d plot
xx1, xx2 = np.meshgrid(np.linspace(X.Weight.min(), X.Weight.max(), 100), 
                       np.linspace(X.ZoomWide.min(), X.ZoomWide.max(), 100))
# plot the hyperplane by evaluating the parameters on the grid
Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, azim=-115, elev=15)

# plot hyperplane
surf = ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y - est.predict(X)
ax.scatter(X[resid >= 0].Weight, X[resid >= 0].ZoomWide, y[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X[resid < 0].Weight, X[resid < 0].ZoomWide, y[resid < 0], color='black', alpha=1.0)

# set axis labels
ax.set_xlabel('Weight')
ax.set_ylabel('ZoomWide')
ax.set_zlabel('Price')
print(est.summary())


# In[101]:


from mpl_toolkits.mplot3d import Axes3D

X = a[['Weight', 'ZoomTelescope']]
y = a['Price']

## fit a OLS model with intercept on Weight and ZoomTelescope
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

## Create the 3d plot 
# Weight/ZoomTelescope grid for 3d plot
xx1, xx2 = np.meshgrid(np.linspace(X.Weight.min(), X.Weight.max(), 100), 
                       np.linspace(X.ZoomTelescope.min(), X.ZoomTelescope.max(), 100))
# plot the hyperplane by evaluating the parameters on the grid
Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, azim=-115, elev=15)

# plot hyperplane
surf = ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y - est.predict(X)
ax.scatter(X[resid >= 0].Weight, X[resid >= 0].ZoomTelescope, y[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X[resid < 0].Weight, X[resid < 0].ZoomTelescope, y[resid < 0], color='black', alpha=1.0)

# set axis labels
ax.set_xlabel('Weight')
ax.set_ylabel('ZoomTelescope')
ax.set_zlabel('Price')
print(est.summary())


# In[ ]:





# In[ ]:





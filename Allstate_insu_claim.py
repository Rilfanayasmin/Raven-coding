# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:06:46 2017

@author: User
"""
#ALL STATES

import pandas as pd
import numpy as  np
import statsmodels.api as sm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.shape
train.head()

#finding the missing value 
train.isnull().sum()
#Analysis the distribution of continuous features
train.describe()

contFeatureslist = []
for colName,x in train.iloc[1,:].iteritems():
    #print(x)
    if(not str(x).isalpha()):
        contFeatureslist.append(colName)
        
print(contFeatureslist)
contFeatureslist.remove("id")
contFeatureslist.remove("loss")

catCount = sum(str(x).isalpha() for x in train.iloc[1,:])
print("Number of categories: ",catCount)

catFeatureslist = []
for colName,x in train.iloc[1,:].iteritems():
    if(str(x).isalpha()):
        catFeatureslist.append(colName)

#Unique categorical values per each category
print(train[catFeatureslist].apply(pd.Series.nunique))

#Convert categorical string values to numeric values
from sklearn.preprocessing import LabelEncoder
for cf1 in catFeatureslist:
    le = LabelEncoder()
    le.fit(train[cf1].unique())
    train[cf1] = le.transform(train[cf1])
train.head(5)

# Univariate analysis for numeric data 

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

## Checking For skewness ###
cont_train = train[contFeatureslist]
plt.figure(figsize=(13,9))
sns.boxplot(data = cont_train )
target_var = train["loss"]
sns.boxplot(data = target_var ) # High Skewness 

train["loss"] = np.log1p(train["loss"]) # done log transform on the value to avoid the skewness of the data
sns.violinplot(data = train,y= "loss")
plt.show()

# Bivariate Analysis ##
############## Correlation #########
contFeatureslist.append("loss")
# finding the highly correlated features in plot
corr_matrix = train[contFeatureslist].corr().abs()
plt.figure(figsize =(13,9))
sns.heatmap(corr_matrix,annot=True)

sns.heatmap(corr_matrix, mask=corr_matrix<0.5,cbar=False,annot=True)
plt.show()

# printing highly correlated features orderd
threshold = 0.5
corr_matrix_list = []
size = 15
for i in range(0,size):
    for j in range(i+1,size):
        if(corr_matrix.iloc[i,j] >= threshold and corr_matrix.iloc[i,j] < 1) or (corr_matrix.iloc[i,j] < 0 and corr_matrix.iloc[i,j] <= -threshold):
            corr_matrix_list.append([corr_matrix.iloc[i,j],i,j])
            
corr_matrix_ordered = sorted(corr_matrix_list,key=lambda x: -abs(x[0]))

for v,i,j in corr_matrix_ordered:
    print("%s and %s = %.2f" % (contFeatureslist[i],contFeatureslist[j],v))

#Data Interaction : scatter plot for highly correlated features

for v,i,j in corr_matrix_ordered:
    sns.pairplot(train,size=6,x_vars=contFeatureslist[i],y_vars=contFeatureslist[j])
    plt.show()
    
# the highly correlated cont11 and cont 12 / also cont1 and cont9 have to remove one safely from each pair

### Categorical Features Analysis ####
    
cols = catFeatureslist
n_cols = 4
n_rows = 29

for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize = (12,8))
    for j in range(n_cols):
        sns.countplot(x=cols[i*n_cols+j], data=train,ax=ax[j])
        
### MODEL FOR TRAIN DATA ###
x = train.iloc[:,1:131].values
y = train['loss']
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
lm = LinearRegression()
lm.fit(x,y)
mse_linear = mean_absolute_error(np.expm1(y),np.expm1(lm.predict(x)))
print(mse_linear)

#### Support vector regression ####
from sklearn import svm
svr = svm.SVR(kernel = 'rbf', C=1).fit(x,y)
mse_svr = mean_absolute_error(np.expm1(y),np.expm1(svr.predict(x)))
print(mse_svr)

train['Y-Pred-svm']=svc.predict(X)
pd.crosstab(train['Y-Pred-svm'],['Loan_Status'])

#### KNN ####
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor()
knn.fit(x,y)

mse_knn = mean_absolute_error(np.expm1(y),np.expm1(knn.predict(x)))
print(mse_knn)

####### Random Forest ##
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=50,max_features=2,oob_score=True)
rf_model.fit(x,y)

mse_rf = mean_absolute_error(np.expm1(y),np.expm1(rf_model.predict(x)))
print(mse_rf)


# The Mean absolute error for the random forest is 541.84 which is lesser than other model.So am using RF method to predict the loss for test data
print(test[catFeatureslist].apply(pd.Series.nunique))

#Convert categorical string values to numeric values
from sklearn.preprocessing import LabelEncoder
for cf1 in catFeatureslist:
    le = LabelEncoder()
    le.fit(test[cf1].unique())
    test[cf1] = le.transform(test[cf1])
test.head(5)

X_test = test.iloc[:,1:131].values
test['loss'] = np.expm1(rf_model.predict(X_test))
submission = pd.DataFrame({"Loss": test["loss"],"ID":test['id']})
submission.to_csv("Sample Submissions.csv", index=False)

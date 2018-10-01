#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:55:03 2018

@author: ericzheng

"""

#dataset: https://archive.ics.uci.edu/ml/datasets/Wine

import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# from matplotlib.colors import ListedColormap
# 
# from scipy.spatial.distance import pdist, squareform
# from scipy import exp
# from scipy.linalg import eigh
# from sklearn.datasets import make_moons
# from sklearn.datasets import make_circles
# =============================================================================


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

#df_wine.head()

#Part 1: Exploratory Data Analysis

#print head and tail of data frame
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 120)
print('\n')
print('Head and Tail:' + '\n')
print(df_wine.head())
print(df_wine.tail())
print('\n')

#1.1 print summary statistics of data frame
pd.set_option('display.width', 100)
print('Summary Statistics:' + '\n')
summary = df_wine.describe()
print(summary)

#1.2 print box plots of data frame

sns.set(rc={'figure.figsize':(18,12)})
sns.boxplot(data=df_wine.iloc[:,1:13])
print('\n' + 'Box Plots:')
plt.show()

sns.set(rc={'figure.figsize':(6,4)})
sns.boxplot(data=df_wine.iloc[:,13:14])
print('\n' + 'Box Plots:')
plt.show()

# =============================================================================
# #
# for i in range(178):
#     #assign color based on class labels
#     if df_wine.iat[i,0] == "1":
#         pcolor = "red"
#     elif df_wine.iat[i,0] == "2":
#             pcolor = "green" 
#     else: 
#             pcolor = "blue"
#       #plot rows of data as if they were series data
#     dataRow = df_wine.iloc[i,1:14]
#     dataRow.plot(color = pcolor)
# plt.xlabel("Attribute Index")
# plt.ylabel(("Attribute Values"))
# #plt.clf()
# plt.figure(figsize=(20,16))
# #plt.tight_layout()
# plt.show()
# =============================================================================

 #1.3 Scatterplot Matrix 

print('\n' + 'Scatterplot Matrix:')
cols = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                    'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
sns.pairplot(df_wine[cols], size=2)
plt.tight_layout()
plt.show()   


# =============================================================================
# #1.3 Heatmap
# corMat = pd.DataFrame(df_wine.corr())
# plt.pcolor(corMat)
# print("\n" + "Heatmap:")
# plt.tight_layout()
# plt.show()
# 
# =============================================================================

#1.4 Correlation Matrix 
print("\n" + "Correlation Matrix:")

sns.set(rc={'figure.figsize':(12,9)})
#sns.set()
cols = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline']
cm = np.corrcoef(df_wine[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
# plt.savefig('images/10_04.png', dpi=300)
plt.show()


# Splitting the data into 80% training and 20% test subsets. random_state = 42

X, y = df_wine.iloc[:, 1:14].values, df_wine.iloc[:, 0].values
 


# =============================================================================
# X = df_wine[['Alcohol', 'Malic acid', 'Ash',
#                     'Alcalinity of ash', 'Magnesium', 'Total phenols',
#                     'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
#                     'Color intensity', 'Hue',
#                     'OD280/OD315 of diluted wines', 'Proline']].values
# y = df_wine['Class label'].values
# =============================================================================

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.2, 
                     stratify=y,
                     random_state=42)

#Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Part 2: Logistic regression classifier v. SVM classifier - baseline

# ### Training a logistic regression model with scikit-learn
#C : float, default: 1.0 Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# =============================================================================
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# =============================================================================

# =============================================================================
# print( classification_report(y_test, y_test_pred, target_names=df_wine.columns = ['Class label']) )
# 
# print( confusion_matrix(y_test, y_test_pred) )
# =============================================================================

print( 'Accuracy Score:' )
print( 'Baseline:' )
#Baseline lr
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1, penalty='l2', random_state=42, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train_std, y_train)

y_train_pred = lr.predict(X_train_std)
y_test_pred = lr.predict(X_test_std)

from sklearn.model_selection import cross_val_score


lr_train_score_cv = np.average(cross_val_score(lr, X_train_std, y_train, cv = 10))
lr_test_score_cv = np.average(cross_val_score(lr, X_test_std, y_test, cv = 10))

print('lr_train_score_cv:', lr_train_score_cv)
print('lr_test_score_cv:',  lr_test_score_cv)

#Baseline SVM
from sklearn.svm import SVC
svm = SVC(C=1.0, kernel='linear', random_state=42)
svm.fit(X_train_std, y_train)

svm_train_score_cv = np.average(cross_val_score(svm, X_train_std, y_train, cv = 10))
svm_test_score_cv = np.average(cross_val_score(svm, X_test_std, y_test, cv = 10))

print('svm_train_score_cv:', svm_train_score_cv)
print('svm_test_score_cv:', svm_test_score_cv)


#PCA transformation
print('\n')
print( 'PCA transformation:' )
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

PCA_X_train = pca.fit_transform(X_train_std)
PCA_X_test = pca.transform(X_test_std)

#PCA lr
lr.fit(PCA_X_train, y_train)
PCA_lr_train_score_cv = np.average(cross_val_score(lr, PCA_X_train, y_train, cv = 10))
PCA_lr_test_score_cv = np.average(cross_val_score(lr, PCA_X_test, y_test, cv = 10))

print('lr_train_score_cv:', PCA_lr_train_score_cv)
print('lr_test_score_cv:',  PCA_lr_test_score_cv )

#PCA SVM
svm.fit(PCA_X_train, y_train)
PCA_svm_train_score_cv = np.average(cross_val_score(svm, PCA_X_train, y_train, cv = 10))
PCA_svm_test_score_cv = np.average(cross_val_score(svm, PCA_X_test, y_test, cv = 10))

print('svm_train_score_cv:', PCA_svm_train_score_cv)
print('svm_test_score_cv:', PCA_svm_test_score_cv)

#LDA transformation
print('\n')
print( 'LDA transformation:' )
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
LDA_X_train = lda.fit_transform(X_train_std, y_train)
LDA_X_test = lda.transform(X_test_std)

#LDA lr
lr.fit(LDA_X_train, y_train)
LDA_lr_train_score_cv = np.average(cross_val_score(lr, LDA_X_train, y_train, cv = 10))
LDA_lr_test_score_cv = np.average(cross_val_score(lr, LDA_X_test, y_test, cv = 10))

print('lr_train_score_cv:', LDA_lr_train_score_cv)
print('lr_test_score_cv:',  LDA_lr_test_score_cv)

#LDA SVM
svm.fit(LDA_X_train, y_train)
LDA_svm_train_score_cv  = np.average(cross_val_score(svm, LDA_X_train, y_train, cv = 10))
LDA_svm_test_score_cv = np.average(cross_val_score(svm, LDA_X_test, y_test, cv = 10))

print('svm_train_score_cv:', LDA_svm_train_score_cv)
print('svm_test_score_cv:', LDA_svm_test_score_cv)

#kPCA transformation (Test several different values for Gamma)
from sklearn.decomposition import KernelPCA
print('\n' + 'kPCA Trasformation:')

kPCA_df = pd.DataFrame()

for g in [0.018, 0.01899, 0.019, 0.0195, 0.02, 0.03, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.09, 0.1, 0.2]:
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
    kPCA_X_train = kpca.fit_transform(X_train_std)
    kPCA_X_test = kpca.transform(X_test_std)
    array = {f: s for f, s in zip()}
    array['gamma'] = g

    #kPCA lr
    lr.fit(kPCA_X_train, y_train)
    kPCA_lr_train_score_cv = np.average(cross_val_score(lr, kPCA_X_train, y_train, cv = 10))
    kPCA_lr_test_score_cv = np.average(cross_val_score(lr, kPCA_X_test, y_test, cv = 10))
    array['lr_train_score_cv'] = kPCA_lr_train_score_cv 
    array['lr_test_score_cv'] = kPCA_lr_test_score_cv

    #kPCA SVM
    svm.fit(kPCA_X_train, y_train)
    kPCA_svm_train_score_cv = np.average(cross_val_score(svm, kPCA_X_train, y_train, cv = 10))
    kPCA_svm_test_score_cv = np.average(cross_val_score(svm, kPCA_X_test, y_test, cv = 10))
    array['svm_train_score_cv'] = kPCA_svm_train_score_cv
    array['svm_test_score_cv'] = kPCA_svm_test_score_cv
    
    kPCA_df=kPCA_df.append(array, ignore_index=True)
    
kPCA_df = kPCA_df.set_index('gamma')[['lr_train_score_cv','lr_test_score_cv', 'svm_train_score_cv', 'svm_test_score_cv']]  

print(kPCA_df)

print("My name is Hao Zheng")
print("My NetID is: haoz7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

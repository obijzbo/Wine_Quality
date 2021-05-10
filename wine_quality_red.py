#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:59:49 2021

@author: tns
"""

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

import visualization as vs

data = pd.read_csv('winequality-red.csv', sep=';')

display(data.head())

print(data.isnull().any())

print(data.info())

#Collect total number of samples
n_wines = data.shape[0]

#Number of wine quality above 6
quality_above_6 = data.loc[(data['quality']>6)]
n_above_6 = quality_above_6.shape[0]

#Number of wine quality below 5
quality_below_5 = data.loc[(data['quality']<5)]
n_below_5 = quality_below_5.shape[0]

#Number of wine quality between 5 and 6
quality_between_5_6 = data.loc[(data['quality']>=5) & (data['quality']<=6)]
n_between_5_6 = quality_between_5_6.shape[0]

#Percentage of each quality
good_percent = n_above_6*100/n_wines
average_percent = n_between_5_6*100/n_wines
insipid_percent = n_below_5*100/n_wines

#Print results
print("Total number of wine data: {}".format(n_wines))
print("Wines with rating 7 and above: {}".format(n_above_6))
print("Wines with rating 5 and 6: {}".format(n_between_5_6))
print("Wines with rating less than 5: {}".format(n_below_5))

print("Percentage of wines with quality 7 and above: {:.2f}%".format(good_percent))
print("Percentage of wines with quality between 5 and 6: {:.2f}%".format(average_percent))
print("Percentage of wines with quality 4 and below: {:.2f}%".format(insipid_percent))

#Pandas describe() is used to view some basic statistical details
display(np.round(data.describe()))

# Visualize skewed continuous features of original data
#vs.distribution(data, 'quality')

pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde');

correlation = data.corr()
# display(correlation)
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")

#Visualize the co-relation between pH and fixed Acidity

#Create a new dataframe containing only pH and fixed acidity columns to visualize their co-relations
fixedAcidity_pH = data[['pH', 'fixed acidity']]

#Initialize a joint-grid with the dataframe, using seaborn library
gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, size=6)

#Draws a regression plot in the grid 
gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})

#Draws a distribution plot in the same grid
gridA = gridA.plot_marginals(sns.distplot)

fixedAcidity_citricAcid = data[['citric acid', 'fixed acidity']]
g = sns.JointGrid(x="fixed acidity", y="citric acid", data=fixedAcidity_citricAcid, size=6)
g = g.plot_joint(sns.regplot, scatter_kws={"s": 10})
g = g.plot_marginals(sns.distplot)

#For each feature find the data points with extreme high or low values
for feature in data.keys():
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(data[feature], q=25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(data[feature], q=75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    interquartile_range = Q3 - Q1
    step = 1.5 * interquartile_range
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])
    
    # OPTIONAL: Select the indices for data points you wish to remove
    outliers = []
    # Remove the outliers, if any were specified
    good_data = data.drop(data.index[outliers]).reset_index(drop = True)

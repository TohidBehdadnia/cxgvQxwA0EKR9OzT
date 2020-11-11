# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:03:36 2020

@author: Tohid
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
##############################################################################

df = pd.read_csv (r'C:\Users\Tohid\Desktop\Project_job\term-deposit-marketing-2020.csv')

cleanup_nums = {"job":     {"blue-collar": 11, "management": 10, "technician": 9,"admin": 8,"services": 7,"retired": 6,"self-employed": 5,"entrepreneur": 4,"unemployed": 3,"housemaid": 2,"student": 1,"unknown": 0},
                "marital": {"married": 1, "divorced": 0, "single": -1}, 
                "education": {"secondary": 2, "tertiary": 3, "primary": 1, "unknown": 0},
                "default": {"no": 0, "yes": 1},
                "housing": {"no": 0, "yes": 1}, 
                "loan": {"no": 0, "yes": 1},
                "contact": {"cellular": 2, "unknown": 1, "telephone": 0},
                "month": {"may": 1, "jul": 2, "aug": 3, "jun": 4, "nov": 5, "apr": 6, "feb": 7, "jan": 8, "mar": 9, "oct": 10, "dec": 11},
                "y": {"no": 0, "yes": 1}}

df.replace(cleanup_nums, inplace=True)
df.head()

dataset=df.values

feature_space=df.values[0:40000,0:13];
target=df.values[0:40000,13];
###############################################################################

min_max_scaler = preprocessing.MinMaxScaler()
feature_space = min_max_scaler.fit_transform(feature_space)

###############################################################################



###############################################################################
   
clf = GaussianNB()

clf.fit(feature_space, target)

###############################################################################


scores = cross_val_score(clf, feature_space, target, cv=5)
print("Average Performance Score.: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

###############################################################################

#
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:11:39 2020

@author: Joshua Benjamin.P
"""

import pandas as pd
import numpy as np
import sklearn
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()
random.seed(42)
file_read = pd.read_csv('datafile_new1.csv')
print(file_read.head(5))
# showing all column names
for col in file_read.columns:
    print(col)

# calculate the unique values in the columns except target
unique_values_Termid = file_read.Termid.unique()
unique_values_Regd = file_read.Regd_No.unique()
unique_values_Course = file_read.Course.unique()
unique_values_CA_100 =  file_read.CA_100.unique()
unique_values_MTT_50 = file_read.MTT_50.unique()
unique_values_ETT_100 = file_read.ETP_100.unique()
unique_values_Course_Att = file_read.Course_Att.unique()
unique_values_MHRDName = file_read.MHRDName.unique()
unique_values_CA_1 = file_read.CA_1.unique()
unique_values_CA_2 = file_read.CA_2.unique()
unique_values_CA_3 = file_read.CA_3.unique()
unique_values_CA_4 = file_read.CA_4.unique()
unique_values_Height = file_read.Height.unique()
unique_values_Weight = file_read.Weight.unique()
unique_values_ScholarType = file_read.ScholarType.unique()
unique_values_Gender = file_read.Gender.unique()
unique_values_Medium = file_read.Medium.unique()
unique_values_CourseType = file_read.CourseType.unique()
unique_values_ProgramType = file_read.ProgramType.unique()

# Calculating unique values for the target data
unique_values_Grade = file_read.Grade.unique()




# 1.) Preprocessing predictors values
copy_dataframe = file_read
# there are four columns where there are missing values and these are
""" 
A.) MTT_50
B.) ETT_100
C.) ETP_100
d.) Course_Att
"""   
# 1.a) preprocessing MTT_50
# Since there is no values in MTT_50 hence this column is dropped
copy_dataframe = copy_dataframe.drop( 'MTT_50', axis=1)
print(copy_dataframe)
# 1.b) preprocessing ETT_100
mean_ETT_100 = copy_dataframe['ETT_100'].mean(skipna = True)
print(mean_ETT_100)
# replacing not filled values in the column ETT_100 with the mean value
copy_dataframe['ETT_100'].fillna(mean_ETT_100, inplace = True)
# print(copy_dataframe)
# 1.c) preprocessing ETP_100
mean_ETP_100 = copy_dataframe['ETP_100'].mean(skipna = True)
print(mean_ETP_100)
# replacing not filled values in the column ETT_100 with the mean value
copy_dataframe['ETP_100'].fillna(mean_ETP_100, inplace = True)
# 1.d) preprocessing Course_Att
mean_Course_Att = copy_dataframe['Course_Att'].mean(skipna = True)
print(mean_Course_Att)
# replacing not filled values in the column Course_Att with the mean value
copy_dataframe['Course_Att'].fillna(mean_Course_Att, inplace = True)

"""
There are some columns which has string value
String value needs to be converted into integer
LabelEncoder will be used for this purpose
"""
copy_dataframe.MHRDName = le.fit_transform(copy_dataframe.MHRDName)
copy_dataframe.ScholarType = le.fit_transform(copy_dataframe.ScholarType)
copy_dataframe.Direction = le.fit_transform(copy_dataframe.Direction)
copy_dataframe.Course = le.fit_transform(copy_dataframe.Course)
copy_dataframe.Gender = le.fit_transform(copy_dataframe.Gender)
copy_dataframe.CourseType = le.fit_transform(copy_dataframe.CourseType)
copy_dataframe.ProgramType = le.fit_transform(copy_dataframe.ProgramType)
copy_dataframe.Medium = le.fit_transform(copy_dataframe.Medium)
copy_dataframe.Grade = le.fit_transform(copy_dataframe.Grade)

"""
Next step is to split dataset into
Predictors Variables
Target Variables
By Convention Predictors Variables are denoted by X
Target Variable by y
"""
df = copy_dataframe
"""
Termid is dropped because it has same value
So it will not be any impact on training
"""
X = df.drop(['Grade','Termid'], axis=1)
y = copy_dataframe['Grade']

"""
Both X and y is in form of dataframe
Conversion of df into numpy array
"""
X = X.values
y = y.values
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)
"""
training data: This is the data with which machine will get trained
testing data: This is the data with which prediction will be done
y_training: Training targets/labels which are grade in this case
y_testing: Testing labels
"""
print("Shape of X_training data:", X_train.shape)
print("Shape of X_testing data:", X_test.shape)
print("Shape of y_training data:", y_train.shape)
print("Shape of y_testing data:", y_test.shape)

# Training and Calculating Score using Decision Tree
tree = DecisionTreeClassifier()
t= tree.fit(X_train, y_train)
ts=t.score(X_test, y_test)
print("Accuracy of Decision tree is {}".format(ts))
# Training and Calculating Score using Random Forest Classifier
forest = RandomForestClassifier()
f = forest.fit(X_train, y_train)
fs = f.score(X_test, y_test)
print("Accuracy of Random Forest is {}".format(fs))
# Training and Calculating Score using Support Vector Machine Classifier
svc = SVC()
s= svc.fit(X_train, y_train)
print("SVC Model Score" , ":" , s.score(X_train, y_train) , "," ,
      "Cross Validation Score" ,":" , s.score(X_test, y_test))

plt.scatter(X[:,0],X[:,1],marker='o',c=y)
plt.show()

import seaborn as sns

y_pred_svc = s.predict(X_test)
print(y_pred_svc)
print(metrics.confusion_matrix(y_test, y_pred_svc, labels=[0, 1, 2, 3]))
print(metrics.classification_report(y_test, y_pred_svc, labels=[0, 1, 2, 3]))
# Final Grade Countplot
plt.figure(figsize=(8,6))
sns.countplot(file_read.Grade, order=['B+', 'A', 'A+', 'C'], palette='Set1')
plt.title('Final Grade - Number of Students',fontsize=20)
plt.xlabel('Final Grade', fontsize=16)
plt.ylabel('Number of Student', fontsize=16)
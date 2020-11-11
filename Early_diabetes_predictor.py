# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:39:27 2020

@author: sasim
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sb
import plotly.express as px

dataset = pd.read_csv("D:\\sasi\\study\\Cheat sheet\\Diabetes-Prediction-Deployment-master\\Diabetes-Prediction-Deployment-master\\Early_diabetes_predictor.csv")

dataset.head()
dataset.info()

desc = dataset.describe()


null = dataset.isnull().sum().sort_values(ascending=False)
na =  dataset.isna().sum().sort_values(ascending=False)

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df = dataset.copy(deep=True)
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

null = df.isnull().sum().sort_values(ascending=False)


# Replacing NaN value by mean, median depending upon distribution
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].meadian(), inplace=True)

#  correlation matrix
sb.heatmap(df.corr(),annot=True)

#To find the correlation of  variable with price by Pairplot
sb.pairplot(df)


# checking distributions using histograms
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)

#countplot & Pie chart for Outcome
fig, ax = plt.subplots(1, 2)
sb.countplot(x='Outcome', data=df, ax=ax[0])
ax[1].pie(dataset['Outcome'].value_counts(), labels=['Positive', 'Negative'], autopct='%1.1f%%')
fig.suptitle('Bar & Pie charts of Outcome of its count & its percentage', fontsize=16)
plt.show()

###x & y
x= dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

# Training & Test 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='l2',C = 1.0,random_state = 0,
                                solver='sag', multi_class='ovr')
log_reg.fit(x_train, y_train)

# Predicting the Test set results
y_pred_log = log_reg.predict(x_test)

# Making the Confusion Matrix & Accuracy& Classification report
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)
classification_report_log = classification_report(y_test,y_pred_log,target_names = ['0','1'])




###knearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,
                        metric='minkowski', p=2)

knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)

# Making the Confusion Matrix & Accuracy& Classification report
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
classification_report_knn = classification_report(y_test,y_pred_knn,target_names = ['0','1'])


###naive bayes

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()

NB.fit(x_train, y_train)
y_pred_NB = NB.predict(x_test)

# Making the Confusion Matrix & Accuracy& Classification report
confusion_matrix(y_test, y_pred_knn)
acc_NB = accuracy_score(y_test, y_pred_NB)
classification_report_NB = classification_report(y_test,y_pred_NB,target_names = ['0','1'])


####Decision Tree_Entropy
from sklearn.tree import DecisionTreeClassifier
DT_E= DecisionTreeClassifier(criterion='entropy')

DT_E.fit(x_train, y_train)

y_pred_DT_E = DT_E.predict(x_test)

# Making the Confusion Matrix & Accuracy& Classification report
confusion_matrix(y_test, y_pred_DT_E)
acc_DT_E = accuracy_score(y_test, y_pred_DT_E)
classification_report_DT_E = classification_report(y_test,y_pred_DT_E,target_names = ['0','1'])

####Decision Tree_gini
from sklearn.tree import DecisionTreeClassifier
DT_G= DecisionTreeClassifier(criterion='gini')

DT_G.fit(x_train, y_train)

y_pred_DT_G = DT_G.predict(x_test)

# Making the Confusion Matrix & Accuracy& Classification report
confusion_matrix(y_test, y_pred_DT_G)
acc_DT_G = accuracy_score(y_test, y_pred_DT_G)
classification_report_DT_G = classification_report(y_test,y_pred_DT_G,target_names = ['0','1'])



###Random Forest entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

RF_E = RandomForestClassifier(n_estimators=20,
                                    criterion='entropy')
RF_E.fit(x_train,y_train)
y_pred_RF_E = RF_E.predict(x_test)

# Making the Confusion Matrix & Accuracy& Classification report
confusion_matrix(y_test, y_pred_RF_E)
acc_RF_E = accuracy_score(y_test, y_pred_RF_E)
classification_report_RF_E = classification_report(y_test,y_pred_RF_E,target_names = ['0','1'])

###Random Forest gini
from sklearn.ensemble import RandomForestClassifier

RF_G = RandomForestClassifier(n_estimators=20,
                                    criterion='gini')
RF_G.fit(x_train,y_train)
y_pred_RF_G = RF_G.predict(x_test)

# Making the Confusion Matrix & Accuracy& Classification report
confusion_matrix(y_test, y_pred_RF_G)
acc_RF_G = accuracy_score(y_test, y_pred_RF_G)
classification_report_RF_G = classification_report(y_test,y_pred_RF_G,target_names = ['0','1'])

#Comparing the accuracy

accuracy={'DT GINI':(acc_DT_G*100),'Log_reg':(acc_log*100) , 'KNN':(acc_knn*100), 'Ran_For_G':(acc_RF_G*100),'DT Entropy':(acc_DT_E*100), 
     'Ran_For_E':(acc_RF_E*100),}
print(accuracy)

#Bar
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(accuracy.keys(),accuracy.values())
plt.show()

#Scatter
plt.scatter(accuracy.keys(),accuracy.values()) 
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

results_table = pd.DataFrame(columns = ['models', 'fpr','tpr','auc'])

predictions = {'LR': y_pred_log, 'KNN': y_pred_knn, 'NB': y_pred_NB, 'DT_E': y_pred_DT_E, 'DT_G': y_pred_DT_G, 
               'RF_E': y_pred_RF_E, 'RF_G': y_pred_RF_G}

for key in predictions:
    fpr, tpr, _ = roc_curve(y_test, predictions[key])
    auc = roc_auc_score(y_test, predictions[key])
    
    results_table = results_table.append({'models': key,
                                         'fpr' : fpr,
                                         'tpr' : tpr,
                                         'auc' : auc}, ignore_index=True)
    
results_table.set_index('models', inplace=True)

print(results_table)

fig = plt.figure(figsize = (8,6))

for i in results_table.index:
    plt.plot(results_table.loc[i]['fpr'], 
             results_table.loc[i]['tpr'], 
             label = "{}, AUC={:.3f}".format(i, results_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color = 'black', linestyle = '--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop = {'size':13}, loc = 'lower right')


plt.show()


###Artficial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



classifier = Sequential()

classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu',input_dim=8))

classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


classifier.fit(x_train,y_train, batch_size=8, epochs=50)

y_pred = classifier.predict(x_test)

y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test,y_pred)
A8 = accuracy_score(y_test,y_pred)






























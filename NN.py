#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import svm
#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_curve, auc
#from itertools import cycle



import pandas as pd
train = pd.read_csv('diabetes.csv')
train.head() #summary
train.info() #more info on data type and if columns have missing values
train.describe() #find mean, std, min, mac, percentile 

train.replace('?', np.nan, inplace = True)
train = train.astype(float)

total = train.isnull().sum().sort_values(ascending=False)
percent =(train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)

mode = pd.Series(train.mode().values[0], index=['Pregnancies', 'Glucose', 'BloodPressure',
'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])
median = train.median()
mean = train.mean()
max = train.max()
min = train.min()
skewness = pd.concat([mode, median, mean, max, min], axis=1, keys=['mode',
'median','mean','max','min'])
skewness 
skewness[skewness.index.isin(['Insulin','SkinThickness','BloodPressure','BMI','Glucose'])]
print(skewness)

# Histogram
missing_value_analysis =
pd.concat([train['Insulin'],train['SkinThickness'],train['BloodPressure'],train['BMI'],train['Glucose']]
 , axis=1, keys=['Insulin', 'SkinThickness','BloodPressure','BMI','Glucose'])
missing_value_analysis.hist(bins=50)

# handle the missing data by applying central measure of tendency
train['BMI'] = train['BMI'].fillna(train['BMI'].median())
train['BloodPressure'] = train['BloodPressure'].fillna(train['BloodPressure'].mean())
train['Glucose'] = train['Glucose'].fillna(train['Glucose'].median())
train['Insulin'] = train['Insulin'].fillna(train['Insulin'].median())
train['SkinThickness'] = train['SkinThickness'].fillna(train['SkinThickness'].mean())


#z-score normalization
X = train
Y = train.pop('Outcome')
X = (train-train.mean())/(train.std())

#train-test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
X.hist(bins=30)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', learning_rate_init=0.001, batch_size=10,
hidden_layer_sizes=(1), random_state=1)
clf.fit(X_train, Y_train)

#Plot the Neural Network. 
print(clf.coefs_)

#To estimate the model's performance, generate predictions on the test dataset.
prediction = clf.predict(X_test)
acc = sum(prediction==Y_test)/len(Y_test)
print("Accuracy for MLP: "+str(acc))

# Confusion matrix for the model.
print(confusion_matrix(Y_test, prediction))
y_score = clf.predict_proba(X_test)[:,1]

#Compute ROC curve and ROC area for each class
fprs = []
tprs = []
roc_aucs = []
fpr, tpr, _ = roc_curve(Y_test, y_score)
roc_auc = auc(fpr, tpr)
fprs.append(fpr)
tprs.append(tpr)
roc_aucs.append(roc_auc)

# SVM
clf = svm.SVC()
clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
acc = sum(prediction==Y_test)/len(Y_test)
print("Accuracy for SVM: "+str(acc))
y_score = clf.fit(X_train, Y_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(Y_test, y_score)
roc_auc = auc(fpr, tpr)
fprs.append(fpr)
tprs.append(tpr)
tprs.append(tpr)
roc_aucs.append(roc_auc)
plt.figure()
lw = 2
colors = cycle(['aqua', 'darkorange'])
labels = ['multi-layer', 'SVM']
for i, color in zip(range(2), colors):
 plt.plot(fprs[i], tprs[i], color=color, lw=lw,
 label='ROC curve of '+labels[i]+' (area = {1:0.2f})'
 ''.format(i, roc_aucs[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()





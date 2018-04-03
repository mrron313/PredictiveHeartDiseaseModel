import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

dataSet = np.genfromtxt("heart_disease_dataset.csv", delimiter="," , missing_values=['?'] ) 

dataSet = dataSet[~np.isnan(dataSet).any(axis=1)]

print("Total Rows : ",len(dataSet))

X = np.array(dataSet)[:,0:13]
y = np.array(dataSet)[:,13]

print("Feature Shape : ", X.shape)
print("Output Shape : ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.12, random_state=42)

print("Train Feature Shape : ", X_train.shape)
print("Test Feature Shape : ", X_test.shape)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test,y_pred)*100
precision_dt = precision_score(y_test, y_pred)*100
recall_dt = recall_score(y_test, y_pred)*100
print("Decision Tree : ")
print("Accuracy : ",accuracy_dt)
print("Precision Score : ",precision_dt)
print("Recall Score : ",recall_dt)
print("\n")

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train,y_train)
y_pred = gnb.predict(X_test)
accuracy_dt = accuracy_score(y_test,y_pred)*100
precision_dt = precision_score(y_test, y_pred)*100
recall_dt = recall_score(y_test, y_pred)*100
print("Naive Bayes : ")
print("Accuracy : ",accuracy_dt)
print("Precision Score : ",precision_dt)
print("Recall Score : ",recall_dt)
print("\n")

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_dt = accuracy_score(y_test,y_pred)*100
precision_dt = precision_score(y_test, y_pred)*100
recall_dt = recall_score(y_test, y_pred)*100
print("Random Forest Classifier : ")
print("Accuracy : ",accuracy_dt)
print("Precision Score : ",precision_dt)
print("Recall Score : ",recall_dt)
print("\n")

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)
accuracy_dt = accuracy_score(y_test,y_pred)*100
precision_dt = precision_score(y_test, y_pred)*100
recall_dt = recall_score(y_test, y_pred)*100
print("Logistic Regression : ")
print("Accuracy : ",accuracy_dt)
print("Precision Score : ",precision_dt)
print("Recall Score : ",recall_dt)

print("\n")
from sklearn import model_selection
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('RFC', RandomForestClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()    

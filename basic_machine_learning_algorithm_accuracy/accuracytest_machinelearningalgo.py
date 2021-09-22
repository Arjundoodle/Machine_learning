import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'/Users/arjunkapoor/Desktop/winequalityN 2.csv')

df.head()
df.shape
df.isnull().sum()
print(df.shape,df.isnull().sum)
df['fixed acidity'] = df['fixed acidity'].fillna(df['fixed acidity'].mean())
df['volatile acidity'] = df['volatile acidity'].fillna(df['volatile acidity'].mean())
df['citric acid'] = df['citric acid'].fillna(df['citric acid'].mode()[0])
df['residual sugar'] = df['residual sugar'].fillna(df['residual sugar'].mode()[0])
df['chlorides'] = df['chlorides'].fillna(df['chlorides'].mean())
df['pH'] = df['pH'].fillna(df['pH'].mean())
df['sulphates'] = df['sulphates'].fillna(df['sulphates'].mean())
X = df.iloc[:,1:12].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0,solver="liblinear")
classifier1.fit(X_train, y_train)


from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, y_train)


from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit(X_train, y_train)

from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf', random_state = 0)
classifier4.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
classifier5 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier5.fit(X_train, y_train)

y_pred1 = classifier1.predict(X_test) 


y_pred2 = classifier2.predict(X_test)

y_pred3 = classifier3.predict(X_test)

y_pred4 = classifier4.predict(X_test)

y_pred5 = classifier5.predict(X_test)

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print("------------------------------------------------------------------------")
print("LOGISTIC REGRESSION")
print(classification_report(y_test, y_pred1))
cm1= confusion_matrix(y_test, y_pred1)
acc1 = accuracy_score(y_test, y_pred1)
print(cm1)
print(acc1)

print("------------------------------------------------------------------------")
print("K-NEAREST NEIGHBOURS")
print(classification_report(y_test, y_pred2))
cm2= confusion_matrix(y_test, y_pred2)
acc2 = accuracy_score(y_test, y_pred2)
print(cm2)
print(acc2)

print("------------------------------------------------------------------------")
print("NAIVE BAYES")
print(classification_report(y_test, y_pred3))
cm3= confusion_matrix(y_test, y_pred3)
acc3 = accuracy_score(y_test, y_pred3)
print(cm3)
print(acc3)

print("------------------------------------------------------------------------")
print("SUPPORT VECTOR MACHINE")
print(classification_report(y_test, y_pred4))
cm4= confusion_matrix(y_test, y_pred4)
acc4 = accuracy_score(y_test, y_pred4)
print(cm4)
print(acc4)

print("------------------------------------------------------------------------")
print("DECISION TREE")
print(classification_report(y_test, y_pred5))
cm5= confusion_matrix(y_test, y_pred5)
acc5 = accuracy_score(y_test, y_pred5)
print(cm5)
print(acc5)
df.hist(figsize=(15,15), layout=(4,4), bins=10)
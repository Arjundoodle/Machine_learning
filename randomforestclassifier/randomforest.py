import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv(r'/Users/arjunkapoor/Desktop/UCI_Credit_Card.csv')
df.head()
df.info()
y=df.iloc[:,24].values
x=df.iloc[:,0:24].values

x_train ,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
classifier1=RandomForestClassifier(random_state = 1,
                                  n_estimators = 750,
                                  max_depth = 15, 
                                  min_samples_split = 1.0,  min_samples_leaf = 1)

classifier1.fit(x_train,y_train)
y_pred1=classifier1.predict(x_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

acc1=accuracy_score(y_test, y_pred1)
con1=confusion_matrix(y_test, y_pred1)
rep1=classification_report(y_test, y_pred1)
print(acc1)
print(rep1)
print(con1)

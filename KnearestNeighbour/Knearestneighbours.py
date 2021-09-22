import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r'/Users/arjunkapoor/Desktop/indian_liver_patient.csv')
m=df['Albumin_and_Globulin_Ratio'].isnull().sum()
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

print(m)

df['Gender'].replace( 'Female',0,inplace=True)
df['Gender'].replace( 'Male',1 ,inplace=True)
print(df.head(5))
df.info()
x=df.iloc[:,0:10].values
y=df.iloc[:,10]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
pred=model.predict(x_test)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

acc1=accuracy_score(y_test, pred)
con1=confusion_matrix(y_test, pred)
rep1=classification_report(y_test, pred)
print(acc1)
print(rep1)
print(con1)


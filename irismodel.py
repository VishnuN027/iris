import pandas as pd
import numpy as np
import pickle
data=pd.read_excel('iriss.xls')
from sklearn.preprocessing import LabelEncoder
label_en=LabelEncoder()
a=['Classification']
for i in np.arange(len(a)):
    data[a[i]]=label_en.fit_transform(data[a[i]])
x=data.drop(['Classification'],axis=1)
y=pd.DataFrame(data['Classification'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=9,metric='minkowski')
knn=classifier.fit(x_train,y_train)
pickle.dump(knn,open('model.pkl','wb') )
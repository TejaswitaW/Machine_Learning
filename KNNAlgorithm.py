#KNN algorithm in python using iris dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
#Loaded required modules

iris=datasets.load_iris()
#loaded iris dataset

print(iris.DESCR)
#printed description on the screen

features=iris.data
labels=iris.target

print(features[0],labels[0])

#I will train classifier
clf=KNeighborsClassifier()

#I will fit classifier
clf.fit(features,labels)
#t=np.array([1,31,12,1])#array of feature
#m=t.reshape(1,-1)#reshaping in numpy,
#if you want to pass one dimensional array then do this


#prediction
##preds=clf.predict([[1,2,3,4]])#requires two d array
preds=clf.predict([[1,2,3,4],[1,5,6,7],[10,11,12,13]])
#You can give multiple features using above
#predicts what will be the labels

print(preds)

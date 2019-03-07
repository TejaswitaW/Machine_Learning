import random
from sklearn import*
import numpy as np
from matplotlib import pyplot as plt
random_indices=np.random.permutation(100)
#Training Set
x_train=x[random_indices[:70]]
y_train=y[random_indices[:70]]
#Validation Set
x_val=x[random_indices[70:85]]
y_val=y[random_indices[70:85]]
#Test Set
x_test=x[random_indices[85:]]
y_test=y[random_indices[85:]]

model=linear_model.LinearRegression()
X=np.matrix(x_train.reshape(len(x_train),1))
                                   
Y=np.matrix(y_train.reshape(len(y_train),1))

model.fit(X,Y)
x_predicted=model.predict(x_test)

plt.plot(x_test,x_predicted)


plt.show()
                                   


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
import random
from sklearn.metrics import mean_squared_error
iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

x=x[:100]
y=y[:100]

number_of_samples=len(y)

random_indices=np.random.permutation(number_of_samples)

num_training_samples=int(number_of_samples*0.7)

x_train=x[random_indices[:num_training_samples]]
y_train=y[random_indices[:num_training_samples]]

num_validation_samples=int(number_of_samples*0.15)

x_val=x[random_indices[num_training_samples:\
                       num_training_samples+num_validation_samples]]
y_val=y[random_indices[num_training_samples:\
                       num_training_samples+num_validation_samples]]

num_test_samples=int(number_of_samples*0.15)
          
x_test=x[random_indices[:num_test_samples:]]
y_test=y[random_indices[:num_test_samples:]]

model=linear_model.LogisticRegression()


full_X=np.concatenate((x_class0,x_class1),axis=0)
full_Y=np.concatenate((y_class0,y_class1),axis=0)

model.fit(full_x,full_y)
          

          

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

d_X=np.array([[1],[2],[3],[11],[13],[87]])
d_X_train=d_X
d_X_test=d_X

d_Y_train=np.array([3,2,5,12,11,10])
d_Y_test=np.array([3,2,5,12,11,10])

model=linear_model.LinearRegression()
model.fit(d_X_train,d_Y_train)
d_Y_predicted=model.predict(d_X_test)

print("Mean Squared error is : ",mean_squared_error(d_Y_test,d_Y_predicted))

print("Weight: ",model.coef_)
print("Intercept: ",model.intercept_)

plt.scatter(d_X_test,d_Y_test)
plt.plot(d_X_test,d_Y_predicted)


plt.show()

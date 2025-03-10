import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  
y = np.array([2, 4, 6, 8, 10])  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.show()
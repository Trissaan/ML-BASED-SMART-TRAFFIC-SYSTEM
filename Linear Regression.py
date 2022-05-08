import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("/Users/jasper/Downloads/Traffic dataset.csv")
print(df.head())
sns.pairplot(df, x_vars=['width', 'area'], y_vars='time', height=4, aspect=1, kind='scatter')
plt.show()
sns.heatmap(df.corr(), annot = True)
plt.show()
x = df.values[:, 0:2]  # get input values from first two columns
y = df.values[:, 2]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
model= LinearRegression()
model.fit(x_train, y_train)
y_pred= model.predict(x_test)
model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
plt.scatter(x_test[:,1],y_test)
plt.scatter(x_test[:,1], y_pred)
plt.xlabel('Area')
plt.ylabel('Time')
plt.title('y_test values vs y_pred values')
plt.show()
meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('R squared: {:.2f}'.format(model.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
width = int(input("Enter the value of width of the lane in feet : "))
area = int(input("Enter the value of area of vehicles occupied in square feet : "))
time = int(model.predict([[width, area]]));
print("The predicted value of duration of green signal is : " + str(int(time)))






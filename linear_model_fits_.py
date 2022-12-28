# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read data from input file
df = np.loadtxt("inputdata7.csv", delimiter=",", dtype=str)

# Slice out the numerical values for the rainfall and productivity columns
rainfall_arr = np.asarray(df[1:, 0], dtype='float64')
productivity_arr = np.asarray(df[1:, 1], dtype='float64')

# Visualize the rainfall vs productivity data
fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
plt.scatter(rainfall_arr, productivity_arr, color="red")
plt.title("Rainfall vs Productivity Trend", fontsize=16)  # Chart title
plt.xlabel("Rainfall", fontsize=14)
plt.ylabel("Productivity", fontsize=14)
plt.show()

# Reshape the rainfall_arr and productivity_arr arrays for use in the model
rainfall = rainfall_arr.reshape(-1, 1)
productivity = productivity_arr.reshape(-1, 1)

# Build a linear regression model using the rainfall and productivity data
model = LinearRegression()
model.fit(rainfall, productivity)

# Get the line of best fit for the model
best_fit = model.predict(rainfall)

# Add the line of best fit to the plot
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
plt.title("Rainfall vs Productivity Trend", fontsize=20)
plt.xlabel("Productivity", fontsize=16)
plt.ylabel("Rainfall", fontsize=16)
plt.scatter(rainfall, productivity, color="red")
plt.plot(rainfall, best_fit, label="Line of Best Fit", color="black")
plt.legend(loc='upper right')
plt.show()

# Estimate productivity for a given rainfall value
predict_value = 310
pred = model.predict([[predict_value]])

# Visualize the estimated productivity value
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
plt.scatter(rainfall, productivity, color="red")  # Rainfall vs productivity data
plt.plot(rainfall, best_fit, label="Line of Best Fit", color="black")  # Line plot for line of best fit
plt.scatter(prediction_value, pred, color="blue", s=40, label="Prediction")  # Estimated value
plt.title("Trend in Rainfall vs Productivity", fontsize=20)
plt.xlabel("Rainfall", fontsize=16)
plt.ylabel("Productivity", fontsize=16)

# Label the estimated value
plt.annotate(pred, (prediction_value, pred), fontsize=15)
plt.legend(loc='upper left')
plt.show()

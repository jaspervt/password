#Libraries
from os import path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import plot_generator


data = pd.read_csv("output.csv")
data.dropna()

# All features
x = data.iloc[:, [3,4,6,7,8,9,10,11,12,13,16,17,20,21,22,23,25]]

# new_strength = 5, new_strength2 = 24
y = data.iloc[:,5:6]


#Normalizing/scaling the data
sc = StandardScaler()
x = sc.fit_transform(x)

# Create encoded arrays for different ratings
# strength 0 = [1. 0. 0. 0. 0.], strength 1 = [0. 1. 0. 0. 0.], strength 3 = [0. 0. 0. 1. 0.], etc.
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# Split train and test data. Train data is 95% of the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.05)


# Train or load the model
if path.exists("ModelRFR.joblib"): # Load the model
    model = load('ModelRFR.joblib') 
    print("Loaded model")
else: # Train the model
    # Create a model object
    model = RandomForestRegressor(n_estimators = 10)

    print("Train model")
    model.fit(x_train,y_train)
    dump(model, "ModelRFR.joblib")
    print("Saves model")

print("Model score:")
print(model.score(x_test, y_test))

# Predict the strength of the test data
y_pred = model.predict(x_test)

# Generate plots and error measurements of the output of the model
plot_generator.generatePlots(y_test, y_pred)

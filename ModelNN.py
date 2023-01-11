#Libraries
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import plot_generator



data = pd.read_csv("output.csv")
data.dropna()

# All features
x = data.iloc[:, [3,4,6,7,8,9,10,11,12,13,16,17,20,21,22,23,25]]
length_x = x.shape[1]

# new_strength = 5, new_strength2 = 24
y = data.iloc[:,5:6]

#Normalizing/scaling the data
sc = StandardScaler()
x = sc.fit_transform(x)

# Create encoded arrays for different ratings
# strength 0 = [1. 0. 0. 0. 0.], strength 1 = [0. 1. 0. 0. 0.], strength 3 = [0. 0. 0. 1. 0.], etc.
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

#split train en test data train_data = 90% van de data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1)


# Train or load the model
if path.exists("ModelNN.h5"): # Load the model
    model = load_model("ModelNN.h5")
    print("Loaded model")
    
else: # Train the model
    # Create a model object
    model = Sequential()
    # input_dim should be the amount of x columns and the last layer must be the amount of outputs you want
    # Define the layers of the neural network
    model.add(Dense(100, input_dim=length_x, activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    #specify loss function and the optimizer
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    print("Train model")
    #amount of epochs
    epochs = 10
    training = model.fit(x_train,y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=128)
    #save model
    model.save("ModelNN.h5")
    print("Saves model")
    train_hist = pd.DataFrame(training.history)
    
    
# Predict the strength of the test data
y_pred = model.predict(x_test)


# Generate plots and error measurements of the output of the model
if 'train_hist' in locals():
    plot_generator.generate_per_epoch(y_test, y_pred, train_hist, epochs)
else:
    plot_generator.generatePlots(y_test, y_pred)


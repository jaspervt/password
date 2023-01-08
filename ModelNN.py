#Libraries
from os import path
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import plot_generator


data = pd.read_csv("output.csv")
data.dropna()

# y = strength(nieuw) |||| x = length, numdigits,num_upper en numlower
# wat voor , is voor rows selecteren na , is voor kolommen
# Alle features
x = data.iloc[:, [3,4,6,7,8,9,10,11,12,13,16,17,20,21,22,23,25]]
print(x)
# print(x.shape)
# test = data.iloc[1:6,[2,3,5,6,7,8,9,10,11,12]]
# print(test)
length_x = x.shape[1]
#1:2 = oude strekte, 11:12 = nieuwe sterkte
y = data.iloc[:,5:6]
# print(y)

#Normalizing/scaling the data
sc = StandardScaler()
x = sc.fit_transform(x)
# print(x.shape)
# print(x)

#hierdoor krijg je array speciaal voor bepaalde rating
#strength 3 = [0. 0. 0. 1. 0.] en strength 1 = [0. 1. 0. 0. 0.] en strength 0 = [1. 0. 0. 0. 0.]
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
# print(y)

#split train en test data train_data = 90% van de data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1)


model = Sequential()
# input_dim moet gelijk aan x (kolommen) en laatste laag moet gelijk aan output die je kan krijgen
#Neuraal netwerk maken
model.add(Dense(100, input_dim=length_x, activation='relu'))
model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
model.add(Dense(3,activation='softmax'))


#specify loss function and the optimizer
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


# print(model.summary())

if path.exists("ModelNN.h5"): # Load the model
    imported_model = load_model("ModelNN.h5")
    print("Loaded model")
else: # Train the model
    print("Train model")
    model.fit(x_train,y_train, validation_data=(x_test, y_test), epochs=1, batch_size=64)
    model.save("ModelNN.h5")
    print("Saves model")

y_pred = model.predict(x_test)


plot_generator.generatePlots(y_test, y_pred)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(10, activation='relu', input_shape=(5,)),
    Dense(1)
])

model.summary()

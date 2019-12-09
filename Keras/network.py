import numpy as np
from keras.utils import np_utils
import tensorflow as tf
tf.python_io = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
# One-hot encoding the output
y = np_utils.to_categorical(y)

# Building the model
xor = Sequential()

# Add required layers
xor.add(Dense(64,input_dim=2))
xor.add(Activation('relu')) # relu似乎很有用，可使准确率达到100%
xor.add(Dense(8))
xor.add(Activation('relu'))
xor.add(Dense(2))
xor.add(Activation('softmax'))

xor.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

xor.summary()

history = xor.fit(X,y,epochs=100,verbose=0)

score = xor.evaluate(X,y)
print("\nAccuracy:",score[-1])

print("\nPredictions:")
print(xor.predict_proba(X))
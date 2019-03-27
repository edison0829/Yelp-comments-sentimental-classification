
# coding: utf-8

# In[1]:

import numpy as np
from keras.utils import np_utils
X_train = np.load('X_train_100d.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test_100d.npy')
y_test = np.load('y_test.npy')
y_train = np_utils.to_categorical(y_train, num_classes=3)
y_test = np_utils.to_categorical(y_test, num_classes=3)


# In[2]:

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


# data pre-processing
#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])   # normalize
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])      # normalize


# Another way to build your neural net
model = Sequential([
    Dense(64, input_dim=100),
    Activation('relu'),
    Dense(3),
    Activation('softmax'),
])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=50)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)


# In[ ]:




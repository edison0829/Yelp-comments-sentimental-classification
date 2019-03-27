
# coding: utf-8

# In[1]:

# limit to only one gpu
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # “0, 1” for multiple

####限制只用一个GPU
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest


# In[2]:

import numpy as np
from keras.utils import np_utils
X_train = np.load('X_train_count.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test_count.npy')
y_test = np.load('y_test.npy')
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)
y_train = np_utils.to_categorical(y_train, num_classes=3)
y_test = np_utils.to_categorical(y_test, num_classes=3)


# In[3]:

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[4]:

# create the model
embedding_vecor_length = 100
model = Sequential()
model.add(Embedding(25000, embedding_vecor_length, input_length=500))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[5]:

test_pred = model.predict(X_test)


# In[7]:

np.save("test_pred_LSTM",test_pred)


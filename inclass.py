from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk import pad_sequence

embedding_size = 128

# Convolution
kernel_size = 5
filters = 64 # number of filters in the cnn
pool_size = 4 # Creates a 4x4 matrix

# LSTM
lstm_output_size = 70


df=pd.read_csv('imdb_master.csv',encoding='latin-1',nrows=50000)

df['label']=df['label'].map({'neg':0,'pos':1})
max_features=6000
# Changing my words into tokens/ Interested in reviews
tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['review'])
list_tokenized_train=tokenizer.texts_to_sequences(df['review'])
max_len=130 # shape of my data,

X_train=pad_sequences(list_tokenized_train,maxlen=max_len)
Y_train=df['label'] # Target data
#
print(X_train.shape,Y_train.shape)

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=max_len))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size)) # Output 70 features
model.add(Dense(1)) #fully connected layer has one output.
model.add(Activation('sigmoid')) # Sigmoid - non linear for complex fitting of data

# Compiling the model, minimize loss as we increase accuracy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Fitting the model on training data
model.fit(X_train,Y_train)
print(X_train.shape)
model.save("model.h5")
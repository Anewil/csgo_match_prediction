import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sqlite3
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
pd.pandas.set_option('display.max_columns', None)

conn = sqlite3.connect('database.db')

data = pd.read_sql_query('SELECT * FROM results', conn)

X = data[['teamRankDifference', 'killsPerRoundDifference',
          'deathsPerRoundDifference', 'assistsPerRoundDifference',
          'entryKillsPerRoundDifference', 'explodesPerRoundDifference',
          'defusesPerRoundDifference', 'eliminationsPerRoundDifference',
          'pistolWinDifference', 'ecoWinDifference', 'forceWinDifference',
          'fullWinDifference']]

y = data['teamOneWinner']

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)

model = Sequential([
    Dense(32, activation='relu', input_shape=(12,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, y_train,
                 batch_size=32, epochs=200,
                 validation_data=(X_val, y_val))

model.evaluate(X_test, y_test)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

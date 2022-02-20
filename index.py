# Import libraries
from logging import ERROR
from re import A, M
from typing import final
from keras import models
from matplotlib import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io import feather_format
import pandas_datareader as data
import streamlit as st
from datetime import date
from tensorflow.python.ops.gen_math_ops import mod

from tensorflow.python.platform.tf_logging import error


today = date.today()
start = today - pd.DateOffset(years=20)
end = today

st.title('Stock Trend Prediction')

# get stock symbol from user
user_input = st.text_input("Enter The Stock Symbol", 'TATAMOTORS.NS')

end = st.text_input("Enter The Date", end)

# get data from yahoo finance
df = data.DataReader(user_input, 'yahoo', start, end)

st.subheader('Data from start to end')

# show data
st.write(df)

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(20,10))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)
ma25 = df.Close.rolling(25).mean()

st.subheader('Closing Price vs Time Chart with 100MA & 75MA & 50MA & 25MA & 10MA')
ma75 = df.Close.rolling(75).mean()
ma50 = df.Close.rolling(50).mean()
ma25 = df.Close.rolling(25).mean()
ma10 = df.Close.rolling(10).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma75)
plt.plot(ma50)
plt.plot(ma25)
plt.plot(ma10)
plt.plot(df.Close)
st.pyplot(fig)

# divide the data into train and test 70 / 30
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
X_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    X_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

#ML Model
# use to create the LSTM model for code reference model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

models = Sequential()

models.add(LSTM(units= 50, activation='relu', return_sequences= True, input_shape = (X_train.shape[1], 1)))
models.add(Dropout(0.2))

models.add(LSTM(units= 60, activation='relu', return_sequences= True))
models.add(Dropout(0.3))

models.add(LSTM(units= 80, activation='relu', return_sequences= True))
models.add(Dropout(0.4))

models.add(LSTM(units= 90, activation='relu', return_sequences= True))
models.add(Dropout(0.5))

models.add(LSTM(units= 100, activation='relu', return_sequences= True))
models.add(Dropout(0.6))

models.add(LSTM(units= 110, activation='relu', return_sequences= True))
models.add(Dropout(0.7))

models.add(LSTM(units= 120, activation='relu', return_sequences= True))
models.add(Dropout(0.8))

models.add(LSTM(units= 130, activation='relu', return_sequences= True))
models.add(Dropout(0.9))

models.add(Dense(units=1))

models.compile(optimizer='adam', loss= 'mean_squared_error')
models.fit(X_train, y_train, epochs= 50)

models.save('Keras_model.h5')
data_testing.head()

#Load my model
from tensorflow import keras
models = keras.models.load_model('Keras_model.h5')


#Testing Part
 
past_500_days = data_testing.tail(500)
final_df = past_500_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = models.predict(x_test)
y_predicted = y_predicted.reshape(y_predicted.shape[0],y_predicted.shape[1])
scaler = scaler.scale_

scaler_factor = 1/scaler[0]
predicted = (y_predicted * scaler_factor).max()
y_test = y_test * scaler_factor

# get prediction
st.subheader('Prediction: ' + str(predicted))

# get original price
st.subheader('Original Price: ' + str(data_testing.tail(1).Close[0]))
e2 = data_testing.tail(1).Close[0]

# get error
st.subheader('Error: ' + str(abs(predicted-e2)))

# get error percentage
st.subheader('Error Percentage: ' + str(abs(predicted-e2)/e2*100))



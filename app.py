from typing import final
from keras import models
from matplotlib import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io import feather_format
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input("Enter The Stock Symbol", 'AAPL')

df = data.DataReader(user_input, 'yahoo', start, end)

st.subheader('Data from 2010 - 2019')

st.write(df.describe())

st.subheader('Closing Prise vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Prise vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Prise vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean() 
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


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
#----

#Load my model
model = load_model('Keras_model.h5')

#Testing Part
 
past_100_days = data_testing.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor


st.subheader('Prediction Vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b' , label = 'Oraginal Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
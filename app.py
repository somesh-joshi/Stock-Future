# Import libraries
from keras import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
from datetime import date, timedelta 

from sklearn.metrics import accuracy_score

today = date.today()
start = today - pd.DateOffset(years=20)
end = today


st.title('Stock Price Prediction')

# get stock symbol from user
user_input = st.text_input("Enter The Stock Symbol", 'TATAPOWER.NS')

end = st.text_input("Enter The Date", end)
# convart end date to datetime
end = pd.to_datetime(end)
start = end - pd.DateOffset(years=20)

# get data from yahoo finance
df = data.DataReader(user_input, 'yahoo', start, end)

start = date(start.year, start.month, start.day)
end = date(end.year, end.month, end.day)
st.subheader('Data from {start} to {end}'.format(start=start, end=end))

# show data
st.write(df)

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

data_testing = pd.DataFrame(df['Close'])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

X_train = []
y_train = []

X_train, y_train = np.array(X_train), np.array(y_train)

#ML Model
# use to create the LSTM model for code reference model

#Load my model
from tensorflow import keras
models = keras.models.load_model('Keras_model.h5')


#Testing Part
 
past_1000_days = data_testing
final_df = past_1000_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = models.predict(x_test)
#accuracy = metrics.accuracy_score(y_true, y_pred)
y_predicted = y_predicted.reshape(y_predicted.shape[0],y_predicted.shape[1])
scaler = scaler.scale_
scaler_factor = 1/scaler[0]
predicted = (y_predicted * scaler_factor)
flat_predicted = predicted.flatten()
ma_predicted = np.sort(flat_predicted)
e1 = ma_predicted.max()
e2 = data_testing.tail(1).Close[0]

st.subheader('Predicted Price Graph')
fig = plt.figure(figsize=(12,6))
plt.plot(ma_predicted)
st.pyplot(fig)


# get prediction
st.subheader('Prediction: ' + str(e1))

# get original price
st.subheader('Original Price: ' + str(e2))

# get error percentage
st.subheader('Error Percentage: ' + str(abs((e1-e2)/e2)*100))

 

# tencer flow a
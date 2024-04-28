import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from plotly import graph_objs as go

# Streamlit UI
st.title('Crypto Price Prediction with LSTM')
# Ambil input simbol saham dari pengguna
#stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL):")
symbol = ('ETH-USD', 'BTC-USD', 'ES=F', 'NQ=F', 'GC=F')
stock_symbol = st.selectbox('Pilih dataset untuk prediksi', symbol)
# Ambil input tanggal dari pengguna
start_date = st.date_input("Select start date", pd.to_datetime('today') - pd.DateOffset(years=2))
end_date = st.date_input("Select end date", pd.to_datetime('today'))

# Ambil input jumlah bulan ke depan dari pengguna menggunakan slider
n_months_forecast = st.slider("Select the number of months for forecast", min_value=1, max_value=12, value=3)

# Download data saham dari tanggal yang dipilih
df = yf.download(stock_symbol, start=start_date, end=end_date)

# Ambil harga penutup ('Close') dan isi nilai yang hilang dengan nilai yang ada sebelumnya
y = df['Close'].fillna(method='ffill')

# Reshape data ke bentuk array 2D dengan satu kolom
y = y.values.reshape(-1, 1)

# Normalisasi data menggunakan MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# Generate the input and output sequences
n_lookback = 60
n_forecast = n_months_forecast * 30  # Setiap bulan dianggap memiliki 30 hari perdagangan

# Prepare dataset and use only Close price value
dataset = df.filter(['Close']).values

# Create len of percentage training set
training_data_len = int(np.ceil((len(dataset) * 90) / 100))

# Scale the dataset between 0 - 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Scaled trained data
train_data = scaled_data[:training_data_len, :]

# Split into trained x and y
x_train = []
y_train = []
for i in range(n_lookback, training_data_len - n_forecast + 1):
    x_train.append(train_data[i - n_lookback:i, 0])
    y_train.append(train_data[i:i + n_forecast, 0])

# Convert trained x and y as numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x trained data as 3 dimension array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Fit the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(25))
model.add(Dense(n_forecast))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

# Generate the forecasts
X_ = y[-n_lookback:]
X_ = X_.reshape(1, n_lookback, 1)

Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

# Calculate MSE and RMSE
mse = mean_squared_error(y[-n_forecast:], Y_)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y_ - y[-n_forecast:]))
mape = np.mean(np.abs((Y_ - y[-n_forecast:]) / y[-n_forecast:])) * 100

# Organize the results in a data frame
df_past = df[['Close']].reset_index()
df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.DateOffset(days=1), periods=n_forecast)
df_future['Forecast'] = Y_.flatten()
df_future['Actual'] = np.nan

results = pd.concat([df_past, df_future]).set_index('Date')  # Menggunakan concat untuk menggabungkan DataFrame

# Tampilkan MSE, RMSE, MAE, dan MAPE
st.write(f'Mean Squared Error (MSE): {mse:.4f}')
st.write(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
st.write(f'Mean Absolute Error (MAE): {mae:.4f}')
st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
# Show Training Data
st.subheader('Training Data And Forecast Data')
st.write(results)

# Show additional information
st.subheader('Additional Information:')
st.write(f'Data Shape: {x_train.shape}')
st.write(f'Size of training set: {x_train.size}')
st.write(f'Number of Samples: {x_train.shape[0]}')
st.write(f'Number of Features: {x_train.shape[1]}')
st.write(f'Number of Forecasts: {n_forecast}')
# Menampilkan informasi jumlah data aktual yang dilatih dan jumlah forecast
st.subheader('Size of Actual and Forecast Data')
st.write(f'Number of Actual Data: {len(df_past)}')
st.write(f'Number of Forecast Data: {len(df_future)}')
# Tampilkan grafik dengan Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results.index, results['Actual'], label='Actual', color='green')
ax.plot(results.index, results['Forecast'], label='Forecast', color='blue')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('Stock Price Prediction with LSTM')
ax.legend()

# Display the Matplotlib figure using Streamlit
st.pyplot(fig)

fig = go.Figure()

# Plot actual data
fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], mode='lines', name='Actual', line=dict(color='green', width=2)))

# Plot forecast data
fig.add_trace(go.Scatter(x=results.index, y=results['Forecast'], mode='lines', name='Forecast', line=dict(color='blue', width=2)))

# Update layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Close Price',
    title='Stock Price Prediction with LSTM'
)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)

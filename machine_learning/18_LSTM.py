import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing

ennusteaika = 12
seqlength = 12

df = pd.read_csv('AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df['Time'] = df.index
#%%
df['PassengersLag'] = df['Passengers'].shift(1)
df['PassengersDiff'] = df.apply(lambda row:
                           row['Passengers']-row['PassengersLag'],axis=1)

for i in range(1,seqlength):
    df['PassengersDiffLag' + str(i)] = df['PassengersDiff'].shift(i)
    
for i in range(1,ennusteaika+1):
    df['PassengersDiffFut' + str(i)] = df['PassengersDiff'].shift(-i)

df_train = df.iloc[:-2*ennusteaika]
df_train.dropna(inplace=True)
df_test = df.iloc[-2*ennusteaika:]

#%%
input_vars = ['PassengersDiff']
for i in range(1,seqlength):
    input_vars.append('PassengersDiffLag'+str(i))
    
output_vars = []
for i in range(1,ennusteaika+1):
    output_vars.append('PassengersDiffFut' + str(i))
    
scaler = preprocessing.StandardScaler()
scalero = preprocessing.StandardScaler()

X = np.array(df_train[input_vars])
X_scaled = scaler.fit_transform(X)
X_scaledLSTM = X_scaled.reshape(X.shape[0], seqlength, 1)
y = np.array(df_train[output_vars])
y_scaled = scalero.fit_transform(y)

X_test = np.array(df_test[input_vars])
X_testscaled = scaler.transform(X_test)
X_testscaledLSTM = X_testscaled.reshape(
    X_test.shape[0],seqlength,1)
#%%
from sklearn import linear_model
modelLR = linear_model.LinearRegression()
XLR = df_train['Time'].values
XLR = XLR.reshape(-1,1)
yLR = df_train['Passengers'].values
yLR = yLR.reshape(-1,1)
modelLR.fit(XLR, yLR)
XLR_test = df_test['Time'].values
XLR_test = XLR_test.reshape(-1,1)
df_test['PassengersAvgPred'] = modelLR.predict(XLR_test)
#%%
slope = modelLR.coef_
#%%
modelLSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(24, input_shape=(seqlength,1),
                        return_sequences=False),
    tf.keras.layers.Dense(ennusteaika)
    ])
modelLSTM.compile(loss='mse',
                  optimizer=tf.optimizers.Adam(0.001),
                  metrics=['mae'])

modelLSTM.fit(X_scaledLSTM, y_scaled, epochs=200, batch_size=seqlength)
#%%
ennusteDiff = scalero.inverse_transform(
    modelLSTM.predict(X_testscaledLSTM[ennusteaika-1].reshape(1,12,1)))

ennuste = np.zeros(13)
ennuste[0] = df_test['Passengers'][df_test.index[ennusteaika-1]]
for j in range(1,13):
    ennuste[j] = ennuste[j-1]+ennusteDiff[0][j-1]+slope

ennuste = np.array(ennuste[1:])
#%%
df_pred = df_test[-12:]
df_pred['PassengersPred'] = ennuste
#%%
plt.plot(df['Month'].values, df['Passengers'].values,
         color = 'black', label='Actual passengers (training)')
plt.plot(df_pred['Month'].values, df_pred['PassengersPred'],
         color='red', label='Prediction')
plt.grid()
plt.legend()
plt.show()
#%%
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(df_pred['Passengers'].values,
                          df_pred['PassengersPred'].values))
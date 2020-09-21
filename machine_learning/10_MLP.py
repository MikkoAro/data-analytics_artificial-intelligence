import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('Titanic.csv')
df.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)

data = [df]
for dataset in data:
    mean = df["Age"].mean()
    std = df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = df["Age"].astype(int)

#print(df['Embarked'].describe())
common_value = 'S'
data = [df]
for dataset in data:
    df['Embarked'] = dataset['Embarked'].fillna(common_value)

df['Sex'] = np.array(pd.get_dummies(df['Sex']))

koodit = {'S': 0, 'Q': 1, 'C': 2}
df['Embarked'] = df['Embarked'].map(koodit)

df_train = df.iloc[np.r_[0:500, 700:len(df)]]
df_test = df[500:700]

X = np.array(df_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
y = np.array(pd.get_dummies(df_train['Survived']))

model = tf.keras.Sequential([
    keras.layers.Dense(10, activation=tf.nn.relu, 
                       input_shape=(X_scaled.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(2, activation = tf.nn.softmax)
    ])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(0.001),
              metrics=['categorical_accuracy'])

model.fit(X_scaled, y, epochs = 30, batch_size = 1)

ennuste = np.argmax(model.predict(X_scaled), axis=1)
df_train['Ennuste'] = ennuste
df_train.drop(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'], axis=1, inplace=True)
# Training data, categorical_accuracy: 0.8509

X = np.array(df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
y = np.array(pd.get_dummies(df_test['Survived']))

model.fit(X_scaled, y, epochs = 30, batch_size = 1)

ennuste = np.argmax(model.predict(X_scaled), axis=1)
df_test['Ennuste'] = ennuste
df_test.drop(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'], axis=1, inplace=True)
# Test data, categorical_accuracy: 0.8800

df_random = df_test.sample(n=20)



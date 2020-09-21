import cv2
import glob
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
train_data = []
train_hot_data = []
files = glob.glob (r"C:\Users\Aro-1\Desktop\Ohjelmistotekniikka\DATA-ANALYTIIKKA_ja_TEKOALY\Koneoppiminen\koirakissakuvat\train\*.jpg")
for myFile in files:
    output = 0
    if myFile.startswith(r'C:\Users\Aro-1\Desktop\Ohjelmistotekniikka\DATA-ANALYTIIKKA_ja_TEKOALY\Koneoppiminen\koirakissakuvat\train\cat'):
        output = 1
    print(myFile)
    image = cv2.imread (myFile)
    scaled_image = cv2.resize(image, (100, 100))
    #grey_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    train_data.append (scaled_image)
    train_hot_data.append(output)
    
training_data = np.array(train_data)

#%%
test_data = []
test_hot_data = []
files = glob.glob (r"C:\Users\Aro-1\Desktop\Ohjelmistotekniikka\DATA-ANALYTIIKKA_ja_TEKOALY\Koneoppiminen\koirakissakuvat\test\*.jpg")
for myFile in files:
    output = 0
    if myFile.startswith(r'C:\Users\Aro-1\Desktop\Ohjelmistotekniikka\DATA-ANALYTIIKKA_ja_TEKOALY\Koneoppiminen\koirakissakuvat\test\cat'):
        output = 1
    print(myFile)
    image = cv2.imread (myFile)
    scaled_image = cv2.resize(image, (100, 100))
    #grey_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    test_data.append (scaled_image)
    test_hot_data.append(output)
    
testing_data = np.array(test_data)

#%%
x_train_flat = training_data/255
x_test_flat = testing_data/255

y_train = np.asarray(pd.get_dummies(train_hot_data))
y_test = np.asarray(pd.get_dummies(test_hot_data))


#%%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=5, activation='relu', input_shape=(100,100,3)),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(128, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='softmax')
    ])

model.compile(loss='binary_crossentropy',
              optimizer=tf.optimizers.Adam(0.001),
              metrics=['categorical_accuracy'])

model.fit(x_train_flat, y_train, validation_data=(x_test_flat, y_test), epochs=10, batch_size=32)

#%%
ennuste_test = model.predict(x_test_flat)

#%%
plt.imshow(x_test_flat[3899])
#%%
oma_data = []
oma_hot_data = []
files = glob.glob (r"C:\Users\Aro-1\Desktop\Ohjelmistotekniikka\DATA-ANALYTIIKKA_ja_TEKOALY\Koneoppiminen\koirakissakuvat\*.jpg")
for myFile in files:
    output = 0
    if myFile.startswith(r'C:\Users\Aro-1\Desktop\Ohjelmistotekniikka\DATA-ANALYTIIKKA_ja_TEKOALY\Koneoppiminen\koirakissakuvat\cat'):
        output = 1
    print(myFile)
    image = cv2.imread (myFile)
    scaled_image = cv2.resize(image, (100, 100))
    #grey_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    oma_data.append (scaled_image)
    oma_hot_data.append(output)
    
oma_data = np.array(oma_data)

oma_data_flat = oma_data/255

y_oma = np.asarray(pd.get_dummies(oma_hot_data))
#%%
plt.imshow(oma_data_flat[0])
#%%
ennuste_oma = model.predict(oma_data_flat)
#%%


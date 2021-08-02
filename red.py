
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Lambda, MaxPooling2D, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
(trainx,trainy),(testx,testy)=mnist.load_data()


trainx = trainx.reshape(60000, 28, 28, 1)
testx = testx.reshape(10000, 28, 28, 1)
trainx=trainx/255
testx = testx / 255 #normalizo 0 a 1
trainy=to_categorical(trainy) #q hago aca?
testy = to_categorical(testy)
num_classes=testy.shape[1]

model = Sequential()
model.add(Convolution2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainx,trainy, validation_data=(testx, testy), epochs=5)


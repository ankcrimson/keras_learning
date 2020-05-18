import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

num_classes = 10

batch_size = 128
epochs = 24

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# deal with different backends
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0],  img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0],  img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# normalize
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 1hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# define the model [usually 1D is for timeseries, 2D for img, 3D for videos etc]
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=batch_size,
                 epochs=epochs, verbose=1, validation_data=(X_test, y_test))

# eavaluate to get scores on test data
score = model.evaluate(X_test, y_test, verbose=0)
print(f'test loss: {score[0]}')
print(f'test accuracy: {score[1]}')


# plot
epoch_list = list(range(1, len(hist.history['acc'])+1))
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()

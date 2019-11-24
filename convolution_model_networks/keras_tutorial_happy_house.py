import os

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model

from convolution_model_networks.utils.kt_utils import *

K.set_image_data_format('channels_last')

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


def happy_model(input_shape):
    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(1, 1))(X_input)
    X = Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Flatten()(X)
    X = Dense(units=1, activation='sigmoid')(X)
    model = Model(inputs=X_input, outputs=X, name='HappyModel')
    return model


if os.path.exists('./models/weights/happyModel.h5'):
    happyModel = load_model('./models/weights/happyModel.h5')
else:
    happyModel = happy_model((64, 64, 3))
    happyModel.compile(optimizer='Adam', loss='binary_crossentropy',
                       metrics=['accuracy'])
    happyModel.fit(X_train, Y_train, batch_size=16,
                   epochs=40, validation_data=(X_test, Y_test),
                   shuffle=True)
    happyModel.save('./models/weights/happyModel.h5')

happyModel.summary()
score = happyModel.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_model(happyModel, to_file='./models/description/happyModel.png', show_shapes=True)

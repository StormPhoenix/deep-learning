import os

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import plot_model

from convolution_model_networks.utils.cnn_utils import load_dataset, convert_to_one_hot

K.set_image_data_format('channels_last')

np.random.seed(2)


def build_model(input_shape):
    x_input = Input(shape=input_shape)
    x = Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding='same')(x_input)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(x)

    x = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same')(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    x = Flatten()(x)
    x = Dense(units=6)(x)
    x = Activation(activation='softmax')(x)

    return Model(inputs=x_input, outputs=x)


CNN_TEST_MODEL_PATH = './models/weights/cnn_test_model.hdf5'
CNN_TEST_MODEL_FINAL_PATH = './models/weights/cnn_test_final_model.hdf5'

if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    index = 6
    plt.imshow(X_train_orig[index])
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))

    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    if os.path.exists(CNN_TEST_MODEL_FINAL_PATH):
        model = load_model(CNN_TEST_MODEL_FINAL_PATH)
    else:
        model = build_model(X_train.shape[1:])
        rmsprop = RMSprop(learning_rate=0.01)
        model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(CNN_TEST_MODEL_PATH, monitor='accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]

        model.fit(X_train, Y_train, batch_size=16, callbacks=callbacks_list,
                  epochs=227, validation_data=(X_test, Y_test), shuffle=True)
        model.save(CNN_TEST_MODEL_FINAL_PATH)

    plot_model(model, to_file='./models/description/cnn_test_model.png', show_shapes=True)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    layer_1_output = Model(inputs=model.input, outputs=model.layers[1].output)
    data = X_train[index]
    data = np.expand_dims(data, 0)
    output = layer_1_output.predict(data)

    for i in range(output.shape[-1]):
        data = output[0, :, :, i]
        im = Image.fromarray(data, mode='L')
        im.save('./models/description/{}.jpg'.format(str(i)))

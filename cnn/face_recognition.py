import numpy as np
import tensorflow as tf
from keras.utils import plot_model

from cnn.utils.fr_utils import load_weights_from_FaceNet, img_to_encoding
from cnn.utils.inception_blocks import faceRecoModel

import keras.backend as K
K.set_image_data_format('channels_first')


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    J(A, P, N) = ∑{[f(A) - f(P)]**2 - [f(A) - f(N)]**2 + a}

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


def build_database(FRmodel):
    database = {}
    database["danielle"] = img_to_encoding("../resources/images/face_net/danielle.png", FRmodel)
    database["younes"] = img_to_encoding("../resources/images/face_net/younes.jpg", FRmodel)
    database["tian"] = img_to_encoding("../resources/images/face_net/tian.jpg", FRmodel)
    database["andrew"] = img_to_encoding("../resources/images/face_net/andrew.jpg", FRmodel)
    database["kian"] = img_to_encoding("../resources/images/face_net/kian.jpg", FRmodel)
    database["dan"] = img_to_encoding("../resources/images/face_net/dan.jpg", FRmodel)
    database["sebastiano"] = img_to_encoding("../resources/images/face_net/sebastiano.jpg", FRmodel)
    database["bertrand"] = img_to_encoding("../resources/images/face_net/bertrand.jpg", FRmodel)
    database["kevin"] = img_to_encoding("../resources/images/face_net/kevin.jpg", FRmodel)
    database["felix"] = img_to_encoding("../resources/images/face_net/felix.jpg", FRmodel)
    database["benoit"] = img_to_encoding("../resources/images/face_net/benoit.jpg", FRmodel)
    database["arnaud"] = img_to_encoding("../resources/images/face_net/arnaud.jpg", FRmodel)
    return database


def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path=image_path, model=model)
    dist = np.linalg.norm(database[identity] - encoding)

    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    return dist, door_open


# GRADED FUNCTION: who_is_it

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    models -- your Inception models instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ### START CODE HERE ###

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


if __name__ == '__main__':
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())
    plot_model(FRmodel, to_file='../resources/images/face_recognition/face_net_desc.png', show_shapes=True)
    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(FRmodel=FRmodel)
    database = build_database(FRmodel=FRmodel)
    verify("../resources/images/face_net/camera_0.jpg", "younes", database, FRmodel)
    verify("../resources/images/face_net/camera_2.jpg", "kian", database, FRmodel)
    who_is_it("../resources/images/face_net/camera_0.jpg", database, FRmodel)

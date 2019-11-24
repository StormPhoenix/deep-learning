from convolution_model_networks.utils.nst_utils import *


def compute_content_cost(a_C, a_G):
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, [-1])
    a_G_unrolled = tf.reshape(a_G, [-1])

    J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)) / (4 * n_H * n_W * n_C)
    return J_content


def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_H*n_W, n_C) (≈2 lines)
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(tf.transpose(a_S))  # notice that the input of gram_matrix is A: matrix of shape (n_C, n_H*n_W)
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.square(GS - GG)) / (4 * n_C ** 2 * (n_W * n_H) ** 2)
    return J_style_layer


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(sess, model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    models -- our tensorflow models
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references models[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the models input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J


if __name__ == '__main__':
    # read image
    # content_image = Image.open('./images/style_transfer/content/louvre.jpg', 'r')
    content_image = plt.imread('./images/style_transfer/content/louvre_small.jpg')
    content_image = reshape_and_normalize_image(content_image)

    # TODO 问下耿协博，VGG16 对输入有什么要求
    style_image = plt.imread('./images/style_transfer/style/monet.jpg', 'r')
    style_image = reshape_and_normalize_image(style_image)

    # TODO 噪声图片
    noise_image = generate_noise_image(content_image)
    save_image('./images/style_transfer/out/noise_image.jpg', noise_image)

    tf.reset_default_graph()

    session = tf.InteractiveSession()
    model = load_vgg_model("./models/weights/style_transfer/imagenet-vgg-verydeep-19.mat")
    print(model)

    # 计算content feature
    session.run(model['input'].assign(content_image))
    out = model['conv4_2']
    content_feature = session.run(out)

    # 定义 content loss
    generate_image_feature = out
    j_content = compute_content_cost(content_feature, generate_image_feature)

    # 计算style loss
    session.run(model['input'].assign(style_image))
    j_style = compute_style_cost(session, model, STYLE_LAYERS)

    # 计算总体损失
    j = total_cost(j_content, j_style)
    optimizer = tf.train.AdamOptimizer(learning_rate=2.0)
    train = optimizer.minimize(j)

    iterations = 200

    session.run(tf.global_variables_initializer())
    session.run(model['input'].assign(noise_image))

    for i in range(iterations):
        session.run(train)
        generated_image = session.run(model['input'])

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = session.run([j, j_content, j_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("./images/style_transfer/out/" + str(i) + ".png", generated_image)

        # save last generated image
    save_image('./images/style_transfer/out/generated_image.jpg', generated_image)

    # content_image = Image.open('./images/style_transfer/louvre.jpg')
    # plt.imshow(content_image)
    # plt.show()

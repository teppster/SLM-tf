import tensorflow as tf
import numpy as np

def simpleLinearModelConstruction(num_input_features, class_num, learning_rate):
    x = tf.placeholder(tf.float32, [None, num_input_features])
    y_true = tf.placeholder(tf.float32, [None, class_num])
    y_true_cls = tf.placeholder(tf.int64, [None])
    W = tf.Variable(tf.zeros([num_input_features, class_num]))
    b = tf.Variable(tf.zeros([class_num]))
    logits = tf.matmul(x, W) + b
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return


def print_accuracy(x, y_true, y_true_cls, data, session, accuracy):
    feed_dict_test = {x: data.test.images,
                      y_true: data.test.labels,
                      y_true_cls: data.test.cls}
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def run(num_iterations, batch_size, data, x, y_true, optimizer):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

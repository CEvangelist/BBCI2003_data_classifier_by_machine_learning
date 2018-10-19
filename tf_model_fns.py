# -*- coding: utf-8 -*-

import tensorflow as tf

# Model Parameters
learning_rate = 1e-4

# Network Parameters
n_classes = 2
dropout_rate = 0.4


# used by estimator
def cnn_model_fn(features, labels, mode):
    import numpy as np
    """Model function for CNN."""

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 50])

    # Convolutional Layer #1
    conv1 = tf.layers.conv1d(
        inputs=input_layer, filters=128, kernel_size=5,
        padding="same", activation=tf.nn.sigmoid)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layers #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1, filters=64, kernel_size=3,
        padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

    # Dense Layer
    pool2_flat = tf.layers.flatten(pool2)
    dense = tf.layers.dense(inputs=pool2_flat, units=2048,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=dropout_rate,  # global parameter: dropout_rate
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout,
                             units=n_classes)  # global parameter, 2 classes

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph, It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate,  # global parameter
        )
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

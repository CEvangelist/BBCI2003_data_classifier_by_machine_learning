# -*- coding: utf-8 -*-

# standard package
import os
# third-party package
import tensorflow as tf
import numpy as np
# module in this repo
from np_train_data_100Hz import read_train_data
from tf_model_fns import cnn_model_fn, H_PARAMS as h_params

SOURCE_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

tf.logging.set_verbosity(tf.logging.DEBUG)

# make and DEBUG Hyper-parameters
h_params.train_steps = 4000
tf.logging.debug(["%s = %s" % (x, y) for x, y in vars(h_params).items()
                  if '__' not in x])


# Application logic below
def main(unused_argv):
    # Load training and eval data
    # 316 epochs, 50 samples, 28 channels, 500ms per epoch
    data_x, data_y, x_min, x_max = read_train_data(
        SOURCE_ROOT_DIR)  # x_min, x_max not used
    data_x = data_x.reshape(-1, 50, 28)
    # data_x = np.rollaxis(data_x, axis=2, start=1)
    train_data = np.asarray(data_x[:200], dtype=np.float32)
    train_labels = np.asarray(data_y[:200], dtype=np.int32)
    eval_data = np.asarray(data_x[200:], dtype=np.float32)
    eval_labels = np.asarray(data_y[200:], dtype=np.int32)

    # Create the Estimator
    model_dir = os.path.join(SOURCE_ROOT_DIR,
                             "TFModels", "bbci_convnet_model")
    bbci_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir, params=h_params)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=h_params.batch_size,
        num_epochs=None,
        shuffle=True)
    bbci_classifier.train(
        input_fn=train_input_fn,
        steps=h_params.train_steps,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = bbci_classifier.evaluate(input_fn=eval_input_fn)
    tf.logging.info("eval_results:\n %s", eval_results)


if __name__ == "__main__":
    tf.app.run()

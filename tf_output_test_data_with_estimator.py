# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from np_train_data_100Hz import read_test_data
from tf_model_fns import cnn_model_fn

SOURCE_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def main(unused_argv):
    output_dir = os.path.join(SOURCE_ROOT_DIR, "outputData",
                              "sp1s_aa_test_result_by_tensorflow.csv")

    model_dir = os.path.join(SOURCE_ROOT_DIR,
                             "TFModels", "bbci_convnet_model")
    test_data_x = read_test_data(SOURCE_ROOT_DIR)

    estimator = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                       model_dir=model_dir)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data_x},
        shuffle=False)

    value_generator = estimator.predict(predict_input_fn)
    for idx, val in enumerate(value_generator):
        print(idx, val)


if __name__ == "__main__":
    tf.app.run()

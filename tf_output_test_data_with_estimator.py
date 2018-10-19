# -*- coding: utf-8 -*-

import os
# third-party package
import pandas as pd
import tensorflow as tf
# module in this repo
from np_train_data_100Hz import read_test_data
from tf_model_fns import cnn_model_fn

SOURCE_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def output_to_csv(generator_, test_data_x, output_dir):
    data_frame = pd.DataFrame(test_data_x)
    pred_labels = []
    for x in generator_:
        pred_labels.append(x["classes"])
    assert len(data_frame) == len(pred_labels)

    data_frame.insert(0, 0, pred_labels, allow_duplicates=True)
    print(data_frame)
    data_frame.to_csv(output_dir, header=False, index=False)


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
    output_to_csv(value_generator, test_data_x, output_dir)


if __name__ == "__main__":
    tf.app.run()

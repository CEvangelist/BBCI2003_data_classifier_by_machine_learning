# -*- coding: utf-8 -*-

import os
# third-party package
import pandas as pd
import tensorflow as tf
# module in this repo
from np_train_data_100Hz import read_test_data
from tf_model_fns import cnn_model_fn, lstm_model_fn

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
    test_data_x = read_test_data(SOURCE_ROOT_DIR)

    model_fns = {
        "0": {
            "model_fn": lstm_model_fn,
            "model_dir": os.path.join(SOURCE_ROOT_DIR,
                                      "TFModels", "bbci_LSTM_model"),
            "output_dir": os.path.join(
                SOURCE_ROOT_DIR, "outputData",
                "sp1s_aa_test_result_by_tf_LSTM.csv")
        },
        "1": {
            "model_fn": cnn_model_fn,
            "model_dir": os.path.join(SOURCE_ROOT_DIR,
                                      "TFModels", "bbci_convnet_model"),
            "output_dir": os.path.join(
                SOURCE_ROOT_DIR, "outputData",
                "sp1s_aa_test_result_by_tf_CNN.csv")
        },
    }

    legal_selection = [x for x in model_fns.keys()]
    for k, v in model_fns.items():
        print("{0}: {1}".format(k, v["model_fn"].__str__().split()[1]))

    selection = str(input("Which model would you like to predict[0]/1: "))
    if selection not in legal_selection:
        print("input key not in legal selection, default to '0'.")
        selection = legal_selection[0]

    selected_fn = model_fns[selection]

    estimator = tf.estimator.Estimator(model_fn=selected_fn["model_fn"],
                                       model_dir=selected_fn["model_dir"])

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data_x},
        shuffle=False)

    value_generator = estimator.predict(predict_input_fn)
    output_to_csv(value_generator, test_data_x, selected_fn["output_dir"])


if __name__ == "__main__":
    tf.app.run()

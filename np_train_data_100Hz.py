# -*- coding: utf-8 -*-

import os
# third-party package
import numpy as np

# SOURCE_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


# 316 epochs, 50 samples, 28 channels, 500ms per epoch
def read_train_data(root_dir):
    data_x = np.ndarray(shape=(316, 1400), dtype=np.float32)
    data_y = np.ndarray((316, 1), dtype=np.float16)

    input_dir = os.path.join(root_dir, "inputData", "sp1s_aa_train.csv")
    with open(input_dir, "rt") as f:
        for row, val in enumerate(f):
            data_x[row] = val.split()[1:]
            data_y[row] = val.split()[0]

    data_y = data_y.astype(np.int32)
    assert data_x.shape[0] == data_y.shape[0]
    x_min = data_x.min()
    x_max = data_x.max()

    return data_x, data_y, x_min, x_max


def read_test_data(root_dir):
    input_dir = os.path.join(root_dir, "inputData", "sp1s_aa_test.csv")
    test_data_x = np.ndarray(shape=(100, 1400), dtype=np.float32)
    with open(input_dir, "rt") as f:
        for i, val in enumerate(f):
            test_data_x[i] = val.split()
    return test_data_x

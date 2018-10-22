
# BBCI2003_data_classifier_by_machine_learning

provide estimated class labels (0 or 1) for every trial of the test data

## Environment & Package requirement

#### Environment

* Windows10 x64
* Python 3.5.6
* Nvidia CUDNN v9.0 if tensorflow-gpu has been installed

#### Packages

* tensorflow or tensorflow-gpu V1.11 or above
* numpy
* pandas
* sklearn

## Data description

See: [desc.md](./inputData/desc.md)

## Usage

* If use sklearn model
  1. Run bbci2003_MLP_by_sklearn.py
  2. A csv file output to ./outputData/sp1s_aa_test_result_by_sklearn.csv
* Else if use tensorflow
  * LSTM model selected:
      1. Run bbci2003_LSTM_by_tensorflow.py to train model
      2. Run tf_output_test_data_with_estimator.py and select LSTM model
      3. A csv file output to ./outputData/sp1s_aa_test_result_by_tf_LSTM.csv
  * CNN model selected:
      1. Run bbci2003_CNN_by_tensorflow.py to train model
      2. Run tf_output_test_data_with_estimator.py and select CNN model
      3. A csv file output to ./outputData/sp1s_aa_test_result_by_tf_CNN.csv

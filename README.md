# BBCI2003_data_classifier_by_machine_learning

provide estimated class labels (0 or 1) for every trial of the test data

## Environment & Package requirement

### Environment

* Windows10 x64
* Python 3.5.6 or later
* Nvidia CUDNN v9.0 or v10.0 if tensorflow-gpu has been installed

### Packages

* tensorflow or tensorflow-gpu V1.11 or later
* numpy
* pandas
* sklearn
* lightgbm

## Data description

See: [desc.md](./inputData/desc.md)

## Usage

* See tf.keras API
  * CNN model:
      1. Just run bbci2003_CNN_by_tensorflow_keras.py (Strongly recommended)
      2. A csv file output to ./outputData/sp1s_aa_test_result_by_tfKeras.csv

## Declaration

* Most of models are deprecated, you are able to see them at branch: [deprecated](https://github.com/CEvangelist/BBCI2003_data_classifier_by_machine_learning/tree/deprecated).

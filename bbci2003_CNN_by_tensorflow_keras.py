#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard package
import os
# third-party package
import tensorflow as tf
import numpy as np
import pandas as pd

SOURCE_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# make and DEBUG Hyper-parameters
EPOCHS = 200
data_shape = (50, 28, 1)

# Load trainset and split into trainset and valset
# 316 epochs, 50 samples, 28 channels, 500ms per epoch
df = pd.read_csv("./inputData/sp1s_aa_train.csv", header=None,
                 delim_whitespace=True)
data_x = df.loc[:, 1:].values
data_x = data_x.reshape(-1, 50, 28, 1)
data_y = df.loc[:, 0].values

train_data = np.asarray(data_x[:200], dtype=np.float32)
train_labels = np.asarray(data_y[:200], dtype=np.int32)
eval_data = np.asarray(data_x[200:], dtype=np.float32)
eval_labels = np.asarray(data_y[200:], dtype=np.int32)

# Build model in functional API
inputs = tf.keras.layers.Input(shape=data_shape)

conv_0 = tf.keras.layers.Conv2D(filters=32, kernel_size=3,)(inputs)
pool_0 = tf.keras.layers.MaxPool2D(pool_size=2)(conv_0)

flat_0 = tf.keras.layers.Flatten()(pool_0)
dense_0 = tf.keras.layers.Dense(units=16)(flat_0)
normalization_0 = tf.keras.layers.BatchNormalization()(dense_0)
dense_1 = tf.keras.layers.Dense(
    units=1,
    activation=tf.keras.activations.sigmoid)(normalization_0)

model = tf.keras.models.Model(inputs=[inputs], outputs=[dense_1])
model.compile(optimizer=tf.keras.optimizers.RMSprop(2e-3),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.binary_accuracy])
model.summary()

model_dir = os.path.join(SOURCE_ROOT_DIR,
                         "TFModels", "bbci_convnet_keras_model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
ckpt_path = os.path.join(model_dir, "ckpt.hdf5")
hist = model.fit(train_data, train_labels, epochs=EPOCHS,
                 callbacks=[
                     tf.keras.callbacks.TensorBoard(log_dir=model_dir),
                     tf.keras.callbacks.ModelCheckpoint(
                         ckpt_path,
                         monitor="val_binary_accuracy",
                         save_weights_only=True,
                         save_best_only=True,
                         verbose=1),
                 ],
                 validation_data=[eval_data, eval_labels],
                 verbose=2,)

model.load_weights(ckpt_path)

test_df = pd.read_csv("./inputData/sp1s_aa_test.csv",
                      header=None, delim_whitespace=True)
test_features = test_df.values.reshape(-1, 50, 28, 1)
test_pred = model.predict(test_features).flatten()

submission = pd.Series(test_pred)
submission.to_csv("./outputData/sp1s_aa_test_result_by_tfKeras.csv",
                  index=False, header=False)

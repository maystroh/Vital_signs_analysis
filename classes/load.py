from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
from keras.utils import Sequence

STEP = 256

import dataset_new

import pandas as pd

class Data_Gen(Sequence):

    def __init__(self, batch_size, preproc, type, x_set, metadata_x_set, y_set):
        self.x, self.metadata_x, self.y = np.array(x_set), np.array(metadata_x_set), np.array(y_set)
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        np.random.shuffle(self.indices)
        # if 'Train' in type:
        #     print(self.indices)
        self.type = type
        self.preproc = preproc
        self.epoch = 1

    def __len__(self):
        # print(self.type + ' - len : ' + str(int(np.ceil(self.x.shape[0] / self.batch_size))))
        # print('length : {}'.format(np.ceil(self.x.shape[0] / self.batch_size)))
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        upper_bound = len(self.indices) if ((idx + 1) * self.batch_size) > len(self.indices)  else (idx + 1) * self.batch_size
        inds = self.indices[idx * self.batch_size:upper_bound]
        # if 'Train' in self.type:
        #     print(' / ' + str(self.epoch) + ' -- ' + str(idx) + ' ' + str(inds))
        batch_x = self.x[inds]
        batch_metadata_x = self.metadata_x[inds]
        batch_y = self.y[inds]
        return self.preproc.process(batch_x, batch_y)

    def on_epoch_end(self):
        self.epoch += 1
        if 'Train' in self.type:
            # print('on epoch end ' + str(self.epoch) + ' ' + self.type)
            np.random.shuffle(self.indices)
            # print(self.indices)

class Data_Gen_optimized(Sequence):

    signal_sampling_frequency = 125

    columns_to_remove = ['patient_file', 'range_selected', 'Devenir ICU', 'DEVENIR J28', 'Devenir Hospitalisation', 'Dataset']

    def __init__(self, batch_size, preproc, type, file):
        self.batch_size = batch_size
        self.type = type
        self.preproc = preproc
        self.interval_time = int(2.048 * self.signal_sampling_frequency)
        self.column_to_classify = 'Devenir ICU'

        df = pd.read_csv(file, sep='|')
        self.preprocessed_df = dataset_new.process_metadata(df, label_encoding_categories=True)
        self.preprocessed_df = self.preprocessed_df[self.preprocessed_df['Dataset'] == type].reset_index(drop=True)

        self.batch_size = batch_size
        self.indices = np.arange(self.preprocessed_df.shape[0])
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.preprocessed_df.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        batch_x = []
        batch_metadata_x = []
        batch_y = []

        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_rows = self.preprocessed_df.loc[inds]
        for index, row in batch_rows.iterrows():

            file_data = np.fromfile(row['patient_file'], dtype='>u2')
            ranges = row['range_selected'].split(":")
            data_to_treat = file_data[int(ranges[0]):int(ranges[1])]

            signal_label = int(row[self.column_to_classify])
            trunc_samp = self.interval_time * int(len(data_to_treat) / self.interval_time)
            batch_x.append(data_to_treat[:trunc_samp])
            batch_y.append([signal_label])

            for x in self.columns_to_remove:
                row.pop(x)

            batch_metadata_x.append(row.values)

        return self.preproc.process(batch_x, batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class Preproc:

    def __init__(self, signals, labels):
        self.signals = signals
        if signals is not None:
            self.mean, self.std = compute_mean_std(signals)
            self.classes = sorted(set(l for label in labels for l in label))
            print('Classes : ' + str(self.classes))

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):

        if self.signals is not None:
            # np.savetxt("x.csv", x, delimiter=",")
            x = (x - self.mean) / self.std
            # np.savetxt("normalized_x.csv", x, delimiter=",")

        x = pad(x, val=0)
        x = x[:, :, None]
        return x

    def process_y(self, y):
        y = np.array(y)
        y = y.reshape(-1)
        # print(y)
        return y

def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    # print(padded)
    for e, i in enumerate(x):
        # if 'train' in type:
        padded[e, : len(i)] = i
        # elif 'gt' in type:
        #     padded[e, :] = [i[0]] * max_len
        # print(padded[e, :])

    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32), np.std(x).astype(np.float32))

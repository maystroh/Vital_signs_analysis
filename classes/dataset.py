import tqdm
import numpy as np
import os
import csv
import pandas as pd

class Dataset:

    #Path of file that contains the dataset
    dataset_path = ''

    #number of values to take for each window (default is 2 seconds)
    interval_time = 256 # To adapt to the code I have : 2 sec = 250 * 8 * 10^(-3) + 6

    column_to_classify = ''
    column_values = []
    is_training=False

    signal_sampling_frequency = 125
    def __init__(self, file_path, column_to_classify, column_values, window_time=2.0, training=False):
        self.interval_time = int(window_time * self.signal_sampling_frequency)
        self.dataset_path = file_path
        self.column_to_classify = column_to_classify
        self.column_values = column_values
        self.is_training = training
        self.df = pd.read_csv(self.dataset_path, sep='|')

    def recursive_len(self, item):
        if type(item) == list:
            return sum(self.recursive_len(subitem) for subitem in item)
        else:
            return 1

    def read_csv_GT(self, type_dataset):
        data_x = []
        data_y = []
        for index, row in tqdm.tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # print(row['patient_file'] + ' ' + row['range_selected'] + ' ' + row[self.column_to_classify] + ' ' + row['Dataset'])
            if row['Dataset'] == type_dataset:
                file_data = np.fromfile(row['patient_file'], dtype='>u2')
                ranges = row['range_selected'].split(":")
                data_to_treat = file_data[int(ranges[0]):int(ranges[1])]

                # segments = []
                signal_label = 1 if (row[self.column_to_classify] == self.column_values[0]) else 0
                # for x in range(0, len(data_to_treat), int(self.interval_time)):
                #     if x + int(self.interval_time) <= len(data_to_treat):
                #         # print(len(data_to_treat[x:x + self.interval_time]))
                #         segments.append(data_to_treat[x:x + self.interval_time])

                # if (len(segments) < 100):
                #     print(row['patient_file'] + ' ' + row['range_selected'])

                trunc_samp = self.interval_time * int(len(data_to_treat) / self.interval_time)
                # print(str(trunc_samp) + ' , ' + str(len(data_to_treat)) + ' , ' + str(len(segments)))
                data_x.append(data_to_treat[:trunc_samp])
                # print(data_to_treat[:trunc_samp])
                # data_y.append([row[self.column_to_classify]] * len(segments))
                data_y.append([signal_label])

            # if len(data_x) > 64:
            #     break

        return data_x, data_y

    def load_training_dataset(self):
        print('*** Loading training dataset ***')
        return self.read_csv_GT('Train')

    def load_dev_dataset(self):
        print('*** Loading dev dataset ***')
        return  self.read_csv_GT('Dev')

    def load_test_dataset(self):
        print('*** Loading testing dataset ***')
        return self.read_csv_GT('Test')



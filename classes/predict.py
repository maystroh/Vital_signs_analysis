from __future__ import print_function

import argparse
import numpy as np
import keras
import os


import load
import util
from dataset import Dataset
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

def predict(model_path):

    classes = ['DECEDE', 'VIVANT']
    dataset_path_testing = "../data/Annotations/signal_annotation_testing_Devenir_Hospitalisation.csv"
    dataset = Dataset(file_path=dataset_path_testing, column_to_classify='Devenir Hospitalisation',
                      column_values=classes,
                      window_time=2.048)  # 256 values to be taken to support the network I have so far.. should be updated later..
    test_dataset = dataset.load_test_dataset()
    preproc = util.load(os.path.dirname(model_path))
    test_x, test_y = preproc.process(*test_dataset)

    model = keras.models.load_model(model_path)
    probs = model.predict(test_x, verbose=1)

    print(probs.shape)
    print(test_y.shape)
    probs = probs.reshape((-1,2))
    test_y = test_y.reshape((-1, 2))
    print(probs.shape)
    print(test_y.shape)
    for i in range(probs.shape[1]):
        fpr, tpr, thresholds = roc_curve(test_y[:, i], probs[:, i])
        area_roc = auc(fpr, tpr)
        print('Area ROC class \'{0}\' : {1}'.format(classes[i], area_roc))

    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs = predict(args.model_path)

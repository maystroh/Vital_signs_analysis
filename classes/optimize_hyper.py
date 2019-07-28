from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import json
import keras
import numpy as np
import os
import random
import time

import network
import load
from dataset import Dataset
import util
import tensorflow as tf
# import talos as ta

MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir, "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")

def fit_with(train, dev, preproc, verbose, **params):

    batch_size = int(params['batch_size'])
    epochs =  int(params['epochs'])
    lr = params['learning_rate']

    model = network.build_network(**params)

    # reduce_lr = keras.callbacks.ReduceLROnPlateau(
    #     factor=0.01,
    #     patience=2,
    #     min_lr=lr)

    path = './logs/run-{0}-{1}-{2}-{3}'.format(batch_size, epochs, lr, params['conv_dropout'])
    tensorboard = keras.callbacks.TensorBoard(log_dir=path, histogram_freq=0,
                              write_graph=True, write_images=False)

    # print(model.summary())
    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *train)
        dev_gen = load.data_generator(batch_size, preproc, *dev)
        print('validation step {0} '.format(int(len(dev[0]) / batch_size)))
        history = model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=epochs,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            callbacks=[tensorboard], verbose=verbose)
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)
        history = model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(dev_x, dev_y),
            callbacks=[tensorboard], verbose=verbose)

    print('history : ' + str(np.mean(history.history['val_acc'])))
    return np.mean(history.history['val_acc'])

def train(args, params):

    print("Loading training set...")

    dataset_path_training = "../data/Annotations/signal_annotation_training_Devenir_Hospitalisation.csv"
    dataset = Dataset(file_path=dataset_path_training, column_to_classify='Devenir Hospitalisation',
                      column_values=['DECEDE', 'VIVANT'], window_time=2.048) #256 values to be taken to support the network I have so far.. should be updated later..
    train = dataset.load_training_dataset()
    dev = dataset.load_dev_dataset()
    print(str(len(train)) + ' ' + str(dataset.recursive_len(train)))
    print(str(len(dev)) + ' ' + str(dataset.recursive_len(dev)))

    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    save_dir = make_save_dir(params['save_dir'], args.experiment)
    util.save(preproc, save_dir)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    from functools import partial

    verbose = 1
    fit_with_partial = partial(fit_with, train, dev, preproc, verbose, **params)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'conv_dropout': (0, 0.5), 'learning_rate': (1e-4, 1e-1), 'batch_size': (8, 32), 'epochs': (5, 20)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    from bayes_opt.observer import JSONLogger
    from bayes_opt.event import Events

    logger = JSONLogger(path="./hypermparam_optimizationlogs_ppg.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    optimizer.maximize(init_points=50, n_iter=10)

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)


if __name__ == '__main__':

    print(tf.VERSION)
    print(keras.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)

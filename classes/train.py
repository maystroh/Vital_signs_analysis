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
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from datetime import datetime

from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve
import uuid

MAX_EPOCHS = 100

def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 4)
    return gbytes

def save_results(aucs_perclass, prs_perclass, epochs, image_name, index):
    from matplotlib.legend_handler import HandlerLine2D
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.figure(index)
    line1, = plt.plot(epochs, aucs_perclass, 'b', label='Dev AUC')
    line2, = plt.plot(epochs, prs_perclass, 'r', label='Dev PRUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Epoch')
    plt.savefig(image_name)

def make_save_dir(dirname, experiment_name, iteration, fold):
    iteration_fold = str(iteration) + '-' + str(fold)
    save_dir = os.path.join(dirname, experiment_name, iteration_fold)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir, "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")

def save_model(model, model_name):
    model_json_conv = model.to_json()
    with open('{}.json'.format(model_name), "w") as json_file:
        json_file.write(model_json_conv)

    model.save_weights('{}.h5'.format(model_name))

from keras import backend as K
from keras.callbacks import Callback
from keras.models import load_model

import warnings
from clr_callback import *
from math import ceil

class ExponentialMovingAverage(Callback):
    """create a copy of trainable weights which gets updated at every
       batch using exponential weight decay. The moving average weights along
       with the other states of original model(except original model trainable
       weights) will be saved at every epoch if save_mv_ave_model is True.
       If both save_mv_ave_model and save_best_only are True, the latest
       best moving average model according to the quantity monitored
       will not be overwritten. Of course, save_best_only can be True
       only if there is a validation set.
       This is equivalent to save_best_only mode of ModelCheckpoint
       callback with similar code. custom_objects is a dictionary
       holding name and Class implementation for custom layers.
       At end of every batch, the update is as follows:
       mv_weight -= (1 - decay) * (mv_weight - weight)
       where weight and mv_weight is the ordinal model weight and the moving
       averaged weight respectively. At the end of the training, the moving
       averaged weights are transferred to the original model.
       """
    def __init__(self, decay=0.999, filepath='temp_weight.hdf5',
                 save_mv_ave_model=True, verbose=0,
                 save_best_only=False, monitor='val_loss', mode='auto',
                 save_weights_only=False, custom_objects={}):
        self.decay = decay
        self.filepath = filepath
        self.verbose = verbose
        self.save_mv_ave_model = save_mv_ave_model
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.custom_objects = custom_objects  # dictionary of custom layers
        self.sym_trainable_weights = None  # trainable weights of model
        self.mv_trainable_weights_vals = None  # moving averaged values
        super(ExponentialMovingAverage, self).__init__()

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs={}):
        self.sym_trainable_weights = self.model.trainable_weights
        # Initialize moving averaged weights using original model values
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                          self.sym_trainable_weights}
        if self.verbose:
            print('Created a copy of model weights to initialize moving'
                  ' averaged weights.')

    def on_batch_end(self, batch, logs={}):
        for weight in self.sym_trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= \
                (1.0 - self.decay) * (old_val - K.get_value(weight))

    def on_epoch_end(self, epoch, logs={}):
        """After each epoch, we can optionally save the moving averaged model,
        but the weights will NOT be transferred to the original model. This
        happens only at the end of training. We also need to transfer state of
        original model to model2 as model2 only gets updated trainable weight
        at end of each batch and non-trainable weights are not transferred
        (for example mean and var for batch normalization layers)."""
        if self.save_mv_ave_model:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best moving averaged model only '
                                  'with %s available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('saving moving average model to %s'
                                  % (filepath))
                        self.best = current
                        model2 = self._make_mv_model(filepath)
                        if self.save_weights_only:
                            model2.save_weights(filepath, overwrite=True)
                        else:
                            model2.save(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving moving average model to %s' % (epoch, filepath))
                model2 = self._make_mv_model(filepath)
                if self.save_weights_only:
                    model2.save_weights(filepath, overwrite=True)
                else:
                    model2.save(filepath, overwrite=True)

    def on_train_end(self, logs={}):
        for weight in self.sym_trainable_weights:
            K.set_value(weight, self.mv_trainable_weights_vals[weight.name])

    def _make_mv_model(self, filepath):
        """ Create a model with moving averaged weights. Other variables are
        the same as original mode. We first save original model to save its
        state. Then copy moving averaged weights over."""
        self.model.save(filepath, overwrite=True)
        model2 = load_model(filepath, custom_objects=self.custom_objects)

        for w2, w in zip(model2.trainable_weights, self.model.trainable_weights):
            K.set_value(w2, self.mv_trainable_weights_vals[w.name])

        return model2

write_validation_files = False

def roc_auc_classes(y_data_test, y_score, nb_classes):
    fpr, tpr, _ = roc_curve(y_data_test, y_score)
    return auc(fpr, tpr)

def pr_auc_classes(y_data_test, y_score, nb_classes):
    return average_precision_score(y_data_test, y_score)

class Metrics(keras.callbacks.Callback):
    def __init__(self, model, preproc, num_classes, train_x, train_y, val_data, step, test_x, test_y, dir_save,
                 model_prefix, iteration, fold):
        self.model = model
        self.X_train = train_x
        self.Y_train = train_y
        self.validation_data = val_data
        self.validation_step = step
        self.X_test = test_x
        self.Y_test = test_y
        self.models_dir = dir_save
        self.model_prefix = model_prefix
        self.iteration_nb = iteration
        self.fold = fold
        self.preprocessing = preproc
        self.num_classes = num_classes
        self.validation_number = 0

    def on_train_begin(self, logs={}):
        self._rocTraindata_perclass = []
        self._prTraindata_perclass = []

        self._rocTestdata_perclass = []
        self._prTestdata_perclass = []

        self._rocdata_perclass = []
        self._rocdata = []
        self._prdata_perclass = []
        self._accdata = []
        self._predictions_test_fold = []

    def on_epoch_end(self, epoch, logs={}):

        import csv
        self.validation_number += 1

        # Eval on train data first
        # train_x, train_y = self.preprocessing.process(self.X_train, self.Y_train)
        # ytrain_predict = self.model.predict(train_x)
        # auc_train_score = roc_auc_classes(train_y, ytrain_predict, self.num_classes)
        # aupr_train_score = pr_auc_classes(train_y, ytrain_predict, self.num_classes)
        # self._rocTraindata_perclass.append(auc_train_score)
        # self._prTraindata_perclass.append(aupr_train_score)

        # Eval on Val data
        GT_list = []
        predictions_list = []
        for batch_index in range(self.validation_step):
            xVal, yVal = next(self.validation_data)

            # for index_batch in range(len(self.validation_data)):
            #     xVal, yVal = self.validation_data[index_batch]
            predictions = self.model.predict(xVal)
            GT_list.extend(yVal)
            predictions_list = predictions_list + predictions.reshape(-1).tolist()

        val_auc_score = roc_auc_classes(GT_list, predictions_list, self.num_classes)
        val_aupr_score = pr_auc_classes(GT_list, predictions_list, self.num_classes)

        # print('Evaluating test data ')
        # Eval on train data first
        test_x, test_y = self.preprocessing.process(self.X_test, self.Y_test)
        ytest_predict = self.model.predict(test_x)
        auc_test_score = roc_auc_classes(test_y, ytest_predict, self.num_classes)
        aupr_test_score = pr_auc_classes(test_y, ytest_predict, self.num_classes)

        # auc_scores = roc_curve_multiclass(y_val, y_predict, y_val.shape[1])
        # auc_mean_classes = np.mean(auc_scores)
        # for index, score in enumerate(auc_scores):
        print(' val_auc ' + str(
            round(val_auc_score, 3)) + ' - val_aupr: ' + str(round(val_aupr_score, 3)))
        # print(' train_auc: ' + str(round(auc_train_score, 3)) + ' - train_aupr: ' + str(round(
        #     aupr_train_score, 3)))
        print(' test_auc: ' + str(round(auc_test_score, 3)) + ' - test_aupr: ' + str(round(
            aupr_test_score, 3)))

        max_val_auc = np.max(self._rocdata_perclass, axis=0) if self._rocdata_perclass else 0
        if val_auc_score > max_val_auc:
            self._predictions_test_fold = ytest_predict

            files = [os.path.join(self.models_dir, i) for i in os.listdir(self.models_dir) if
                     os.path.isfile(os.path.join(self.models_dir, i)) and '{}-{}-{}'.format(self.model_prefix,
                                                                                            self.iteration_nb,
                                                                                            self.fold) in i]
            [os.remove(file) for file in files]
            weights_architecture_save_path = os.path.join(self.models_dir,
                                                          '{}-{}-{}-{}-{}-{}'.format(self.model_prefix,
                                                                                     self.iteration_nb,
                                                                                     self.fold, epoch + 1,
                                                                                     round(val_auc_score, 3),
                                                                                     round(auc_test_score, 3)))
            save_model(self.model, weights_architecture_save_path)

        self._rocdata_perclass.append(val_auc_score)
        self._prdata_perclass.append(val_aupr_score)

        self._rocTestdata_perclass.append(auc_test_score)
        self._prTestdata_perclass.append(aupr_test_score)

        return

    def get_rocdata(self):
        return self._rocdata

    def get_prdata_perclass(self):
        return self._prdata_perclass

    def get_accdata(self):
        return self._accdata

    def get_rocdata_perclass(self):
        return self._rocdata_perclass

    def get_prTraindata_perclass(self):
        return self._prTraindata_perclass

    def get_rocTraindata_perclass(self):
        return self._rocTraindata_perclass

    def get_prTestdata_perclass(self):
        return self._prTestdata_perclass

    def get_rocTestdata_perclass(self):
        return self._rocTestdata_perclass

    def get_predictions_test_fold(self):
        return self._predictions_test_fold

model_uuid = str(uuid.uuid4())[:5]

def train(args, params):

    files = os.listdir("./gt/")
    launch_nb_times = 2

    print("######## Launch times the training and validation #########".format(launch_nb_times))
    max_val_auc_values = []
    max_test_auc_values = []

    for iteration in range(1, launch_nb_times + 1):

        best_val_auc = []
        best_test_auc_for_val = []

        predictions_test_folds = np.array([])
        gt_test_folds = np.array([])
        print("\n **** Performing cross validation on dataset.. iteration {} ****** ".format(iteration))

        for index, file in enumerate(files):

            print("Loading dataset set... {0} *********".format(file))
            dataset_path = "./gt/{0}".format(file)
            dataset = Dataset(file_path=dataset_path, column_to_classify='Devenir ICU',
                              column_values=['VIVANT', 'DECEDE'], window_time=2.048, training=True) #256 values to be taken to support the network I have so far.. should be updated later..

            train = dataset.load_training_dataset()
            dev = dataset.load_dev_dataset()
            test = dataset.load_test_dataset()
            gt_test_folds = test[1] if index == 0 else np.vstack((gt_test_folds, test[1]))

            print('Training samples :' + str(len(train[0])))
            print('Dev samples :' + str(len(dev[0])))

            print("Building preprocessor...")
            preproc = load.Preproc(*train)
            save_dir = make_save_dir(params['save_dir'], args.experiment, iteration, index+1)
            print(save_dir)
            util.save(preproc, save_dir)

            num_classes = 1 if True else len(preproc.classes)
            params.update({
                "input_shape": [None, 1],
                "num_categories": num_classes
            })

            print('*********** Building network with {0} pooling options'.format(params['pooling']))
            model, run_metadata = network.build_network(**params)

            # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            # stopping = keras.callbacks.EarlyStopping(patience=100)

            import psutil
            class MyCallback(keras.callbacks.Callback):
                def on_batch_end(self, batch, logs):
                    process = psutil.Process(os.getpid())
                    print("   Memory percent : " + str(process.memory_percent()))

            # checkpointer = keras.callbacks.ModelCheckpoint(
            #     filepath=str(get_filename_for_saving(save_dir)),
            #     save_best_only=False)

            batch_size = params.get("batch_size", 32)

            path = './logs/run-{0}-{1}'.format(datetime.now().strftime("%b %d %Y %H:%M:%S"),index+1)
            tensorboard = keras.callbacks.TensorBoard(log_dir=path, histogram_freq=0,
                                                      write_graph=True, write_images=False)

            if index == 0:
                print(model.summary())
                print("Model memory needed for batchsize {0} : {1} Gb".format(batch_size, get_model_memory_usage(batch_size, model)))

            if params.get("generator", False):
                train_gen = load.data_generator(batch_size, preproc, 'Train', *train)
                dev_gen = load.data_generator(batch_size, preproc, 'Dev', *dev)

                valid_metrics = Metrics(model, preproc, num_classes, train[0], train[1], dev_gen, len(dev[0]) // batch_size, test[0], test[1],
                                        save_dir, model_uuid, iteration, index + 1)

                step_cyclic_lr = 2. * ceil(len(train[0]) / batch_size + 1 if len(train[0]) % batch_size != 0 else len(train[0]) // batch_size)
                clr_triangular = CyclicLR(base_lr=params['learning_rate'] * 10e-3, max_lr=params['learning_rate'],
                                          step_size=step_cyclic_lr,
                                          mode='exp_range', gamma=0.9997)

                model.fit_generator(
                    train_gen,
                    steps_per_epoch=len(train[0]) / batch_size + 1 if len(train[0]) % batch_size != 0 else len(train[0]) // batch_size,
                    epochs=MAX_EPOCHS,
                    validation_data=dev_gen,
                    validation_steps=len(dev[0]) / batch_size + 1  if len(dev[0]) % batch_size != 0 else len(dev[0]) // batch_size,
                    callbacks=[valid_metrics, MyCallback(), clr_triangular, tensorboard])

                # train_gen = load.Data_Gen(batch_size, preproc, 'Train', *train)
                # dev_gen = load.Data_Gen(batch_size, preproc, 'Dev', *dev)

                # train_gen = load.Data_Gen_optimized(batch_size, preproc, 'Train', file)
                # dev_gen = load.Data_Gen_optimized(batch_size, preproc, 'Dev', file)
                # valid_metrics = Metrics(dev_gen, len(dev[0]) // batch_size, batch_size)
                # # del train
                # # del dev
                # model.fit_generator(
                #     train_gen,
                #     epochs=MAX_EPOCHS,
                #     validation_data=dev_gen,
                #     callbacks=[moving_average_decay, valid_metrics, MyCallback(), checkpointer, reduce_lr, tensorboard])


            else:
                train_x, train_y = preproc.process(*train)
                print( '*** preprocessing already done **** ')
                print(train_x.shape)
                print(train_y.shape)

                network.log_output_layers(model, train_x)
                dev_x, dev_y = preproc.process(*dev)

                model.fit(
                    train_x, train_y,
                    batch_size=batch_size,
                    epochs=MAX_EPOCHS,
                    validation_data=(dev_x, dev_y),
                    callbacks=[tensorboard])

            aucs_val_perclass = np.array(valid_metrics.get_rocdata_perclass())
            print(aucs_val_perclass)
            max_val_auc = np.max(aucs_val_perclass)
            best_val_auc.append(max_val_auc)
            max_val_index = np.where(aucs_val_perclass == max_val_auc)

            # aucs_train_perclass = np.array(valid_metrics.get_rocTraindata_perclass())
            # train_auc = aucs_train_perclass[max_val_index]

            aucs_test_perclass = np.array(valid_metrics.get_rocTestdata_perclass())
            test_auc = aucs_test_perclass[max_val_index]
            best_test_auc_for_val.append(np.max(test_auc))

            # Compute the AUC and mAP for all test predictions at once
            predictions_test_folds = valid_metrics.get_predictions_test_fold() if index == 0 else np.vstack(
                (predictions_test_folds, valid_metrics.get_predictions_test_fold()))

        auc_test_score = roc_auc_classes(gt_test_folds, predictions_test_folds, gt_test_folds.shape[1])
        aupr_test_score = pr_auc_classes(gt_test_folds, predictions_test_folds, gt_test_folds.shape[1])

        mean_best_val_auc = np.mean(best_val_auc)
        mean_best_test_auc_for_val = np.mean(best_test_auc_for_val)

        max_val_auc_values.append(mean_best_val_auc)
        max_test_auc_values.append(auc_test_score)

        print("*** Mean Best Val AUC found {}".format(round(mean_best_val_auc, 3)))
        print(" With test AUC : {} / on all predictions at once : {} ".format(round(mean_best_test_auc_for_val, 3),
                                                                              round(auc_test_score , 3)))

    print("***************** Mean Val AUC over {0} iterations is {1}".format(launch_nb_times, str(
        round(np.mean(max_val_auc_values), 3))))
    print("***************** Mean Test AUC over {0} iterations is {1}".format(launch_nb_times, str(
        round(np.mean(max_test_auc_values), 3))))

    from matplotlib.legend_handler import HandlerLine2D
    from matplotlib.font_manager import FontProperties

    import matplotlib
    matplotlib.use('agg')

    import matplotlib.pyplot as plt

    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
    font0 = FontProperties()

    launch_times = range(1, launch_nb_times + 1)
    plt.figure(1)
    line1, = plt.plot(launch_times, max_val_auc_values, label='Val AUC')
    plt.plot(launch_times, max_test_auc_values, label='Test AUC')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Az')
    plt.xlabel('Iteration')

    plt.text(500, 300, 'Mean Val={}'.format(str(round(np.mean(max_val_auc_values), 3))),
             ha='center', va='center',
             transform=None
             )

    plt.text(500, 315, 'Mean Test={}'.format(str(round(np.mean(max_test_auc_values), 3))),
             ha='center', va='center',
             transform=None
             )

    plt.savefig('test_vs_val_AUC.png')


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

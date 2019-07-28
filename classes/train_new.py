from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import keras
import numpy as np
import os
import json
import network
import load
from dataset_new import Dataset
import util
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.utils.vis_utils import plot_model
from datetime import datetime

from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve
import uuid

MAX_EPOCHS = 200
TENSORBOARD_KEYWORD = 'NATURECNN_MAXPOOL_DATAAUGM'

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

from keras.callbacks import TensorBoard
from keras import backend as K

def make_image(tensor):
    import matplotlib.pyplot as plt
    width, height = 960, 720
    plt.figure()
    x_data = [x * 8 * pow(10, -3) for x in list(range(0, len(tensor)))]
    plt.plot(x_data, tensor)
    plt.title("Normalized signal")

    # figure_signal.outline_line_width = 5
    # figure_signal.outline_line_alpha = 0.3
    # figure_signal.outline_line_color = "#FF0000" if gt == 'DECEDE' else "#00FF00"

    import io
    output = io.BytesIO()
    plt.savefig(output, format='png')
    image_string = output.getvalue()
    output.close()
    plt.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=3,
                            encoded_image_string=image_string)

class TrainTensorBoard(TensorBoard):
    def __init__(self, crossval_index, signal_train_x, dev_train_x, preproc, log_dir='./logs', **kwargs):
        self.preproc = preproc
        self.signals_training= signal_train_x
        self.signals_dev = dev_train_x
        self.log_dir_training = '{0}/{1}-{2}-{3}'.format(log_dir, TENSORBOARD_KEYWORD, datetime.now().strftime("%b %d %H:%M:%S"), crossval_index + 1)
        super(TrainTensorBoard, self).__init__(self.log_dir_training, **kwargs)

    def set_model(self, model):
        # Setup writer for validation metrics
        self.writer = tf.summary.FileWriter(self.log_dir_training)
        super(TrainTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):

        indices_train_patient = np.random.randint(low=0, high=len(self.signals_training), size=(10,))

        indices_dev_patient = np.random.randint(low=0, high=len(self.signals_dev), size=(5,))

        training_signals = np.take(self.signals_training, indices_train_patient, axis=0)
        dev_signals = np.take(self.signals_dev, indices_dev_patient, axis=0)

        training_signals_prepoc = np.squeeze(self.preproc.process_x(training_signals), axis=2)
        dev_signals_prepoc = np.squeeze(self.preproc.process_x(dev_signals), axis=2)

        for index_train in range(0, len(training_signals_prepoc)):

            signal_train = training_signals_prepoc[index_train]
            image = make_image(signal_train)
            summary = tf.Summary(value=[tf.Summary.Value(tag='Training examples', image=image)])
            self.writer.add_summary(summary, epoch)

            if index_train < len(dev_signals_prepoc):
                signal_dev = dev_signals_prepoc[index_train]
                image = make_image(signal_dev)
                summary = tf.Summary(value=[tf.Summary.Value(tag='Development examples', image=image)])
                self.writer.add_summary(summary, epoch)

        self.writer.flush()

        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super(TrainTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainTensorBoard, self).on_train_end(logs)
        self.writer.close()

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
        self.X_train, self.Y_train = preproc.process(train_x, train_y)
        self.validation_data = val_data
        self.validation_step = step
        self.X_test = test_x
        self.Y_test = test_y
        self.X_test, self.Y_test = preproc.process(test_x, test_y)
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
        ytrain_predict = self.model.predict(self.X_train)
        auc_train_score = roc_auc_classes(self.Y_train, ytrain_predict, self.num_classes)
        aupr_train_score = pr_auc_classes(self.Y_train, ytrain_predict, self.num_classes)
        self._rocTraindata_perclass.append(auc_train_score)
        self._prTraindata_perclass.append(aupr_train_score)

        # Eval on Val data
        GT_list = []
        predictions_list = []
        # for batch_index in range(self.validation_step):
        #     xVal, yVal = next(self.validation_data)

        for index_batch in range(len(self.validation_data)):
            xVal, yVal = self.validation_data[index_batch]
            predictions = self.model.predict(xVal)
            GT_list.extend(yVal)
            predictions_list = predictions_list + predictions.reshape(-1).tolist()

        val_auc_score = roc_auc_classes(GT_list, predictions_list, self.num_classes)
        val_aupr_score = pr_auc_classes(GT_list, predictions_list, self.num_classes)

        # print('Evaluating test data ')
        # Eval on train data first
        ytest_predict = self.model.predict(self.X_test)
        auc_test_score = roc_auc_classes(self.Y_test, ytest_predict, self.num_classes)
        aupr_test_score = pr_auc_classes(self.Y_test, ytest_predict, self.num_classes)

        # auc_scores = roc_curve_multiclass(y_val, y_predict, y_val.shape[1])
        # auc_mean_classes = np.mean(auc_scores)
        # for index, score in enumerate(auc_scores):
        print(' val_auc ' + str(
            round(val_auc_score, 3)) + ' - val_aupr: ' + str(round(val_aupr_score, 3)))
        print(' train_auc: ' + str(round(auc_train_score, 3)) + ' - train_aupr: ' + str(round(
            aupr_train_score, 3)))
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

        logs.pop('val_acc')
        logs.pop('acc')
        logs.update({'train_AUC': round(auc_train_score, 3), 'test_AUC': round(auc_test_score, 3),
                     'val_AUC': round(val_auc_score, 3)})

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

    # from tensorflow.python.client import device_lib
    #
    # local_device_protos = device_lib.list_local_devices()
    # print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    launch_nb_times = 1
    perform_data_augmentation = True
    is_training = True
    window_duration = 2.048 #2.048 seconds : equivalent to 256 numerical value

    print("######## Launch times the training and validation for {0} times ######### \n with data_augmentation: {1} & windows size {2}".format(launch_nb_times,
                                                                                                                                               perform_data_augmentation ,
                                                                                                                                               window_duration))
    max_val_auc_values = []
    max_test_auc_values = []

    files = os.listdir("./gt/")
    for iteration in range(1, launch_nb_times + 1):

        best_val_auc = []
        best_test_auc_for_val = []

        predictions_test_folds = np.array([])
        gt_test_folds = np.array([])
        print("\n **** Performing cross validation on dataset.. iteration {} ****** ".format(iteration))

        for index, file in enumerate(files):

            print("Loading dataset set... {0} *********".format(file))
            dataset_path = "./gt/{0}".format(file)
            dataset_instance = Dataset(file_path=dataset_path, column_to_classify='Devenir ICU',
                              column_values=['VIVANT', 'DECEDE'], window_time=window_duration, training=is_training, augment_data=perform_data_augmentation) #256 values to be taken to support the network I have so far.. should be updated later..

            signal_train_x, metadata_train_x, train_y = dataset_instance.load_training_dataset()
            signal_dev_x, metadata_dev_x, dev_y = dataset_instance.load_dev_dataset()
            signal_test_x, metadata_test_x, test_y = dataset_instance.load_test_dataset()

            print('Training data contains zeros : {0} '.format(np.all(np.array(signal_train_x) == 0)))
            # Generate random data to test the network
            # signal_train_x = np.random.randint(low=1500, high=3000, size=(len(signal_train_x), len(signal_train_x[0]))).tolist()
            # signal_dev_x = np.random.randint(low=1500, high=3000, size=(len(signal_dev_x), len(signal_dev_x[0]))).tolist()
            # signal_test_x = np.random.randint(low=1500, high=3000, size=(len(signal_test_x), len(signal_test_x[0]))).tolist()

            print('Training samples :' + str(len(signal_train_x)))
            print('Dev samples :' + str(len(signal_dev_x)))
            print('Test samples :' + str(len(signal_test_x)))

            gt_test_folds = test_y if index == 0 else np.vstack((gt_test_folds, test_y))

            print("Building preprocessor...")
            preproc = load.Preproc(signal_train_x, train_y)
            save_dir = make_save_dir(params['save_dir'], args.experiment, iteration, index+1)
            print(save_dir)
            util.save(preproc, save_dir)

            num_classes = 1 if True else len(preproc.classes)
            params.update({
                "input_shape": [None, 1],
                "num_categories": num_classes
            })

            # keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "7910bigpu:7004"))

            print('*********** Building network with {0} pooling options'.format(params.get('pooling', False)))
            model, run_metadata = network.build_network(**params)
            # model, run_metadata = network.build_network_from_scratch(**params)
            # model, run_metadata = network.build_resNet_model(**params)

            if index == 0:
                print(model.summary())
                log_output_model = True
                if log_output_model:
                    train_x, train_y = preproc.process(signal_train_x, train_y)
                    network.log_output_layers(model, train_x)
                    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

            batch_size = params.get("batch_size", 32)
            tensorboard = TrainTensorBoard(index, signal_train_x, signal_dev_x, preproc)

            train_gen = load.Data_Gen(batch_size, preproc, 'Train', signal_train_x, metadata_train_x, train_y)
            dev_gen = load.Data_Gen(batch_size, preproc, 'Dev', signal_dev_x, metadata_dev_x, dev_y)

            valid_metrics = Metrics(model, preproc, num_classes, signal_train_x, train_y, dev_gen,
                                    np.ceil(len(signal_dev_x) / batch_size),
                                    signal_test_x, test_y,
                                    save_dir, model_uuid, iteration, index + 1)

            callbacks = [valid_metrics, tensorboard]

            if params.get('use_cycle_lr', False):
                from clr_callback import CyclicLR
                from math import ceil
                step_cyclic_lr = 2. * ceil(
                    len(signal_train_x) / batch_size + 1 if len(signal_train_x) % batch_size != 0 else len(
                        signal_train_x) // batch_size)
                clr_triangular = CyclicLR(base_lr=params['learning_rate'] * 10e-3, max_lr=params['learning_rate'],
                                          step_size=step_cyclic_lr,
                                          mode='exp_range', gamma=0.9994)
                callbacks.append(clr_triangular)
            # train_gen = load.Data_Gen_optimized(batch_size, preproc, 'Train', file)
            # dev_gen = load.Data_Gen_optimized(batch_size, preproc, 'Dev', file)

            print('steps_per_epoch {}'.format(np.ceil(len(signal_train_x) / batch_size)))
            print('validation_steps {}'.format(np.ceil(len(signal_dev_x) / batch_size)))

            model.fit_generator(
                train_gen,
                steps_per_epoch=np.ceil(len(signal_train_x) / batch_size),
                validation_steps=np.ceil(len(signal_dev_x) / batch_size),
                epochs=MAX_EPOCHS,
                validation_data=dev_gen,
                callbacks=callbacks, verbose=1)

            aucs_val_perclass = np.array(valid_metrics.get_rocdata_perclass())
            # print(aucs_val_perclass)
            max_val_auc = np.max(aucs_val_perclass)
            best_val_auc.append(max_val_auc)
            max_val_index = np.where(aucs_val_perclass == max_val_auc)

            aucs_train_perclass = np.array(valid_metrics.get_rocTraindata_perclass())
            train_auc = aucs_train_perclass[max_val_index]

            aucs_test_perclass = np.array(valid_metrics.get_rocTestdata_perclass())
            test_auc = aucs_test_perclass[max_val_index]
            best_test_auc_for_val.append(np.max(test_auc))

            # "conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],

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

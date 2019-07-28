
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation
from keras.optimizers import RMSprop
from keras import backend as K
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from keras.datasets import mnist

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def get_model(input_shape, dropout2_rate=0.5):
    """Builds a Sequential CNN model to recognize MNIST.

    Args:
      input_shape: Shape of the input depending on the `image_data_format`.
      dropout2_rate: float between 0 and 1. Fraction of the input units to drop for `dropout_2` layer.

    Returns:
      a Keras model

    """
    # Reset the tensorflow backend session.
    # tf.keras.backend.clear_session()
    # Define a CNN model to recognize MNIST.
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,
                     name="conv2d_1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
    model.add(Dropout(0.25, name="dropout_1"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation='relu', name="dense_1"))
    model.add(Dropout(dropout2_rate, name="dropout_2"))
    model.add(Dense(num_classes, activation='softmax', name="dense_2"))
    return model

def fit_with(input_shape, verbose, dropout2_rate, lr, batch_size, epochs):

    # Create the model using a specified hyperparameters.
    model = get_model(input_shape, dropout2_rate)

    # Train the model for a specified number of epochs.
    optimizer = RMSprop(lr=lr)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    batch_size = int(batch_size)
    epochs = int(epochs)
    # Train the model with the train dataset.
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Evaluate the model with the eval dataset.
    score = model.evaluate(x_test, y_test, steps=10, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Return the accuracy.
    return score[1]

from functools import partial

verbose = 1
fit_with_partial = partial(fit_with, input_shape, verbose)

from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'dropout2_rate': (0.05, 0.5), 'lr': (1e-5, 1e-1), 'batch_size': (8, 64), 'epochs': (5, 30)}

optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.maximize(init_points=50, n_iter=10)


for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)
from numpy import loadtxt

dataset = loadtxt("a.csv", delimiter=",")
x = dataset[:,0:8]
y = dataset[:,8]

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def diabetes():
    
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=100, batch_size=10, verbose=0)
    
    return model

from keras.activations import relu, elu

# add input parameters to the function
def diabetes(x_train, y_train, x_val, y_val, params):
    
    # replace the hyperparameter inputs with references to params dictionary 
    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=8, activation='elu'))
    #model.add(Dense(8, activation=params['activation']))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # make sure history object is returned by model.fit()
    out = model.fit(x, y,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    #validation_split=.3,
                    verbose=0)
    
    # modify the output model
    return out, model

import talos as ta

p = {
	'first_neuron': (10,50, 1),
    'epochs': (50,100, 1),
    'batch_size': (2, 128, 2),
    'lr': (0.0001, 0.1, 0.01)
}

t = ta.Scan(x, y, p, diabetes, dataset_name='test_talos', experiment_no='2', print_params=True, \
            reduction_method='correlation' , reduction_interval=50, reduction_window=20, reduction_threshold=0.2, reduction_metric='val_acc')

# r = ta.Reporting("breast_cancer_1.csv")
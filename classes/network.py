from keras import backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

def auroc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, num_thresholds=10000, curve='ROC', summation_method='careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def aupr(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, num_thresholds=10000, curve='PR', summation_method='careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D
    from keras.regularizers import l2
    if params.get('use_weight_decay', False):
        layer = Conv1D(
            filters=num_filters,
            kernel_size=filter_length,
            strides=subsample_length,
            padding='same',
            kernel_initializer=params["conv_init"], kernel_regularizer = l2(params["conv_weight_decay"]))(layer)
    else:
        layer = Conv1D(
            filters=num_filters,
            kernel_size=filter_length,
            strides=subsample_length,
            padding='same',
            kernel_initializer=params["conv_init"])(layer)

    # print(layer)
    return layer

def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer, params["conv_dropout"], **params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add 
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length, padding='same')(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
    # print('pad: {0} / shortcut {1} / {2} / {3}'.format(zero_pad, shortcut.input_shape, shortcut.output_shape))

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(layer, dropout=params["conv_dropout"] if i > 0 else 0, **params)
        layer = add_conv_weight( layer, params["conv_filter_length"], num_filters, subsample_length if i == 0 else 1, **params)


    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index( index, params["conv_num_filters_start"], **params)
        # print('{0} : {1} / {2}'.format(index, num_filters, subsample_length))
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer

def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, CuDNNLSTM, Dropout
    from keras.layers.wrappers import TimeDistributed
    from keras.regularizers import l2

    # HASSAN : recently added to support binary classification
    if not params['rnn_layers'] and params['pooling'] is '':
        layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    else:
        if params['rnn_layers']:
            layer = CuDNNLSTM(64)(layer)
            layer = Dropout(0.5)(layer)
            # layer = CuDNNLSTM(32)(layer)
            # layer = Dropout(0.5)(layer)
        else:
            if params['pooling'] == 'avg':
                layer = GlobalAveragePooling1D()(layer)
            elif params['pooling'] == 'max':
                layer = GlobalMaxPooling1D()(layer)

        if params.get('use_weight_decay', False):
            layer = Dense(params["num_categories"],
                          kernel_regularizer=l2(params["dense_weight_decay"]))(layer)
        else:
            layer = Dense(params["num_categories"])(layer)

    return Activation('softmax' if params['num_categories'] != 1 else 'sigmoid')(layer)

def add_compile(model, **params):
    from keras.optimizers import Adam
    if params.get('use_cycle_lr', False):
        optimizer = Adam(clipnorm=params.get("clipnorm", 1))
    else:
        optimizer = Adam(lr=params.get("learning_rate"), clipnorm=params.get("clipnorm", 1))

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    model.compile(loss='categorical_crossentropy' if params['num_categories'] != 1 else 'binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'], options=run_options, run_metadata=run_metadata)
    return run_metadata

def build_network( **params):
    from keras.models import Model
    from keras.layers import Input, Masking

    if params.get('use_mask_layer', False):
        inputs = Masking(mask_value=params['mask_value'],input_shape=params['input_shape'])
    else:
        inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    print(params)
    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)

    output = add_output_layer(layer, **params)
    model = Model(inputs=[inputs], outputs=[output])

    if params.get("compile", True):
        metadata = add_compile(model, **params)
    return model, metadata

#####################################
## Model definition              ##
## ResNet based on Rajpurkar    ##
##################################

def build_resNet_model(**params):
    from keras.utils.vis_utils import plot_model
    from keras.models import Model
    from keras.layers import BatchNormalization, Input,  Conv1D, Dropout, Dense, MaxPooling1D, Activation, add, \
        TimeDistributed, Lambda, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D
    import numpy as np

    # Add CNN layers left branch (higher frequencies)

    k = 1    # increment every 4th residual block
    p = True # pool toggle every other residual block (end with 2^8)
    convfilt = 64
    convstr = 1
    ksize = 16
    poolsize = 2
    poolstr  = 2
    drop = 0.5

    # Modelling with Functional API
    input1 = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')

    ## First convolutional block (conv,BN, relu)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(input1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(drop)(x1)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x)
    # Merge both branches
    x = add([x1, x2])
    del x1,x2

    ## Main loop
    p = not p
    for l in range(15):

        if (l%4 == 0) and (l>0): # increment k on every fourth residual block
            k += 1
             # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt*k,kernel_size=1)(x)
        else:
            xshort = x
        # Left branch (convolutions)
        # notice the ordering of the operations has changed
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)
        if p:
            x1 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x1)

        # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(xshort)
        else:
            x2 = xshort  # pool or identity
        # Merging branches
        x = add([x1, x2])
        # change parameters
        p = not p # toggle pooling

    # Final bit
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = GlobalMaxPooling1D()(x)
    x = GlobalAveragePooling1D()(x)

    # x = Lambda(lambda y: K.batch_flatten(y), output_shape=x_shape)(x)
    #x = Dense(1000)(x)
    #x = Dense(1000)(x)
    out = Dense(params['num_categories'], activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=out)

    if params.get("compile", True):
        metadata = add_compile(model, **params)

    return model, metadata

def build_network_from_scratch( **params):
    from keras.models import Sequential
    from keras.layers import GlobalAveragePooling1D, CuDNNLSTM,  Conv1D, Dropout, Dense, MaxPooling1D, Lambda

    print(params)

    model = Sequential()
    # model.add(Conv1D(filters=64, kernel_size=16, activation='relu', input_shape=(None, 1)))
    # model.add(Conv1D(filters=64, kernel_size=16, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Lambda(lambda x: K.batch_flatten(x)))
    model.add(CuDNNLSTM(100, input_shape=(None, 1)))
    # model.add(GlobalAveragePooling1D())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params['num_categories'], activation='sigmoid'))

    if params.get("compile", True):
        metadata = add_compile(model, **params)

    return model, metadata

def log_output_layers(model, input_data):

    inp = model.input  # input placeholder
    outputs = []
    for layer in model.layers:
        if 'inputs' not in layer.name:
            outputs.append(layer.output)

        # if 'add_4' in layer.name:
        #     break

    functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function
    layer_outs = functor([input_data[:1, :], 1.])
    #logging outputs
    for counter, layer in enumerate(outputs):
        print("{0} : {1}".format(layer, layer_outs[counter].shape))
        if 'Sigmoid' in layer.name:
            import numpy as np
            print(layer.name + ' -> ' + str(layer_outs[counter]))
            # np.savetxt("layerout.csv", layer_outs[counter][0], delimiter=",")

def LSTM_CNN_SANDWICH(input_shape,output_shape):
    K.set_image_data_format('channels_last')
    n_outputs = output_shape
    input_shape = input_shape
    
    input           = Input(shape=input_shape)
    lstm_1          = Bidirectional(LSTM(32,return_sequences=True))(input)
    activation_1    = Activation("relu")(lstm_1)
    reshape_layer_1 = Reshape((activation_1.shape[1],activation_1.shape[2],1))(activation_1)
    cnn_1           = Conv2D(64, (5,1), strides=(2,1), activation='relu')(reshape_layer_1)
    max_pool_1      = MaxPooling2D((2,1), strides=(2,1))(cnn_1)
    cnn_1_2         = Conv2D(32, (5,1), strides=(2,1), activation='relu')(max_pool_1)
    max_pool_1_2    = MaxPooling2D((2,1), strides=(2,1))(cnn_1_2)
    #global_pool    = GlobalAveragePooling2D()(cnn_1_2)
    batch_norm_1    = BatchNormalization()(max_pool_1_2)

    dropout_2       = Dropout(0.6)(batch_norm_1)
    
    reshape_layer_2 = Reshape((-1,batch_norm_1.shape[3]))(dropout_2)
    lstm_2          = Bidirectional(LSTM(32,return_sequences=True))(reshape_layer_2)
    activation_2    = Activation("relu")(lstm_2)
    reshape_layer_3 = Reshape((activation_2.shape[1],activation_2.shape[2],1))(activation_2)
    cnn_2           = Conv2D(64, (5,1), strides=(2,1), activation='relu')(reshape_layer_3)
    max_pool_2      = MaxPooling2D((2,1), strides=(2,1))(cnn_2)
    cnn_2_2         = Conv2D(32, (5,1), strides=(2,1), activation='relu')(max_pool_2)
    #max_pool_2_2   = MaxPooling2D((2,1), strides=(2,1))(cnn_2_2)
    global_pool_2   = GlobalAveragePooling2D()(cnn_2_2)
    batch_norm_2    = BatchNormalization()(global_pool_2)

    dropout_3       = Dropout(0.6)(batch_norm_2)

    #global_avg      = GlobalAveragePooling2D()(batch_norm_2)

    dense           = Dense(n_outputs)(dropout_3 )
    activation_3    = Activation("softmax")(dense)

    model = Model(inputs = input, outputs= activation_3)
    model.summary()

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy',f1_m])

    return model
    
def LSTM_CNN(input_shape=None,n_outputs=None):
    
    K.set_image_data_format('channels_last')
    n_outputs = n_outputs
    input_shape = input_shape
    input           = Input(shape=input_shape)

    lstm_1          = Bidirectional(LSTM(32,return_sequences=True))(input)
    activation_1    = Activation("relu")(lstm_1)
    lstm_2          = Bidirectional(LSTM(32,return_sequences=True))(activation_1)
    activation_2    = Activation("relu")(lstm_2)

    reshape_layer_1 = Reshape((lstm_2.shape[1],lstm_2.shape[2],1))(activation_2)
    cnn_1           = Conv2D(64, (5,5), strides=(2,2), activation='relu')(reshape_layer_1)
    max_pool_1      = MaxPooling2D((2,2), strides=(2,2))(cnn_1)
    cnn_2           = Conv2D(128, (3,3), strides=(1,1), activation='relu')(max_pool_1)
    global_avg      = GlobalAveragePooling2D()(cnn_2)
    dense           = Dense(n_outputs,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.005))(global_avg)

    model = Model(inputs = input, outputs= dense)
    model.summary()

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy',f1_m])
    
    
    return model

def ConvLSTM_PARALLEL(input_shape=None,n_outputs=None):
    
    K.set_image_data_format('channels_last')
    input_shape = input_shape
    input           = Input(shape=input_shape)

    lstm_1          = Bidirectional(LSTM(32,return_sequences=True))(input)
    activation_1    = Activation("relu")(lstm_1)
    activation_1    = Dropout(0.3)(activation_1)
    lstm_2          = Bidirectional(LSTM(32,return_sequences=True))(activation_1)
    activation_2    = Activation("relu")(lstm_2)
    reshape_layer_1 = Reshape((lstm_2.shape[1],lstm_2.shape[2],1))(activation_2)
    
    cnn_1           = Conv2D(64, (5,5), strides=(2,2), activation='relu')(reshape_layer_1)
    max_pool_1      = MaxPooling2D((2,2), strides=(2,2))(cnn_1)
    cnn_2           = Conv2D(128, (3,3), strides=(1,1), activation='relu')(max_pool_1)
    global_avg      = GlobalAveragePooling2D()(cnn_2)
    flatten         = Flatten()(global_avg)

    reshape_2         = Reshape((input.shape[1],input.shape[2],1))(input)
    cnn_1_2           = Conv2D(64, (5,5), strides=(2,2), activation='relu')(reshape_2)
    #max_pool_1_2     = MaxPooling2D((2,2), strides=(2,2))(cnn_1_2)
    cnn_2_2           = Conv2D(128, (3,3), strides=(1,1), activation='relu')(cnn_1_2)
    global_avg_2      = GlobalAveragePooling2D()(cnn_2_2)

    reshape_layer_1_2 = Reshape((global_avg_2.shape[-1],1))(global_avg_2)
    reshape_layer_1_2 = Dropout(0.3)(reshape_layer_1_2)
    lstm_1_2          = Bidirectional(LSTM(32,return_sequences=True))(reshape_layer_1_2)
    activation_1_2    = Activation("relu")(lstm_1_2)
    lstm_2_2          = Bidirectional(LSTM(32,return_sequences=True))(activation_1_2)
    activation_2_2    = Activation("relu")(lstm_2_2)
    flatten_2         = Flatten()(activation_2_2)

    concatenate       = Concatenate()([flatten,flatten_2])
    dropout_3         = Dropout(0.6)(concatenate)
    dense             = Dense(n_outputs,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.005))(dropout_3)



    model = Model(inputs = input, outputs= dense)
    model.summary()

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy',f1_m])
    
    return model

def LSTM_CNN_PARALLEL(input_shape=None,n_outputs=None):
    
    K.set_image_data_format('channels_last')
    input_shape = input_shape
    input           = Input(shape=input_shape)

    lstm_1          = Bidirectional(LSTM(32,return_sequences=True))(input)
    activation_1    = Activation("relu")(lstm_1)
    lstm_2          = Bidirectional(LSTM(32,return_sequences=True))(activation_1)
    activation_2    = Activation("relu")(lstm_2)
    lstm_3          = Bidirectional(LSTM(32))(activation_2)
    activation_3    = Activation("relu")(lstm_3)
    activation_3    = Dropout(0.5)(activation_3)
    flatten         = Flatten()(activation_3)


    reshape_2         = Reshape((input.shape[1],input.shape[2],1))(input)
    cnn_1_2           = Conv2D(64, (5,1), strides=(2,2), activation='relu')(reshape_2)
    #max_pool_1_2     = MaxPooling2D((2,2), strides=(2,2))(cnn_1_2)
    cnn_2_2           = Conv2D(128, (3,3), strides=(1,1), activation='relu')(cnn_1_2)
    global_avg_2      = GlobalAveragePooling2D()(cnn_2_2)
    global_avg_2      = Dropout(0.5)(global_avg_2)
    flatten_2         = Flatten()(global_avg_2)

    concatenate       = Concatenate()([flatten,flatten_2])
   
    dense             = Dense(n_outputs,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.005))(concatenate)



    model = Model(inputs = input, outputs= dense)
    model.summary()

    rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy',f1_m])
    
    
    return model

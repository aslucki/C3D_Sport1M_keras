from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D


def create_model_sequential():
    """ Creates model object with the sequential API:
    https://keras.io/models/sequential/
    """

    model = Sequential()
    input_shape = (16, 112, 112, 3)

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', name='conv1',
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    return model


def create_model_functional():
    """ Creates model object with the functional API:
     https://keras.io/models/model/
     """
    inputs = Input(shape=(16, 112, 112, 3,))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu',
                   padding='same', name='conv1')(inputs)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                         padding='valid', name='pool1')(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu',
                   padding='same', name='conv2')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool2')(conv2)

    conv3a = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3a')(pool2)
    conv3b = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3b')(conv3a)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool3')(conv3b)

    conv4a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4a')(pool3)
    conv4b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4b')(conv4a)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool4')(conv4b)

    conv5a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5a')(pool4)
    conv5b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5b')(conv5a)
    zeropad5 = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)),
                             name='zeropad5')(conv5b)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool5')(zeropad5)

    flattened = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flattened)
    dropout1 = Dropout(rate=0.5)(fc6)

    fc7 = Dense(4096, activation='relu', name='fc7')(dropout1)
    dropout2 = Dropout(rate=0.5)(fc7)

    predictions = Dense(487, activation='softmax', name='fc8')(dropout2)

    return Model(inputs=inputs, outputs=predictions)


def create_features_exctractor(C3D_model, layer_name='fc6'):
    extractor = Model(inputs=C3D_model.input,
                      outputs=C3D_model.get_layer(layer_name).output)
    return extractor

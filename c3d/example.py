from sport1m_model import create_model_functional
import numpy as np


def main():
    model = create_model_functional()
    try:
        model.load_weights('models/C3D_Sport1M_weights_keras_2.2.4.h5')
    except OSError as err:
        print('Check path to the model weights\' file!\n\n', err)

    # 16 black frames with 3 channels
    dummy_input = np.zeros((1, 16, 112, 112, 3))

    prediction_softmax = model.predict(dummy_input)
    predicted_class = np.argmax(prediction_softmax)

    print('{}Success, predicted class index is: {}{}'.format('\033[92m',
                                                             predicted_class,
                                                             '\033[0m'))


if __name__ == "__main__":
    main()

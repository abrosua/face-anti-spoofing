from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Concatenate


def generate_model(param: str, shape: Tuple[int, int]):
    """
    Handle function to instantiate the model.
    :param param: The path of the saved model's weights
    :param shape: Shape of the desired input.
    :return: TF model of the Face Classiier
    """

    img_height, img_width = shape
    pretrain_net = mobilenet_v2.MobileNetV2(input_shape = (img_width, img_height, 3),
                                            include_top = False,
                                            weights = 'imagenet')

    # Adding extra layer for our problem
    x = pretrain_net.output
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Dropout(rate=0.2, name='extra_dropout1')(x)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(units=128, activation='relu', name='extra_fc1')(x)
    # x = Dropout(rate=0.2, name='extra_dropout1')(x)
    x = Dense(1, activation='sigmoid', name='classifier')(x)

    model = Model(inputs=pretrain_net.input, outputs=x, name='mobilenetv2_spoof')
    model.load_weights(param)
    model.trainable = False  # Freeze all the layer (inference only)
    print(f"Loading weights from: '{param}'")

    return model


if __name__ == "__main__":
    pass

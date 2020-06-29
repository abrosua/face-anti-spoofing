import os
from typing import Optional, Tuple

import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.layers import Conv2D, Dense, Dropout
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
                                            weights = None)

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


class SaveModel:
    def __init__(self, model, savedir: str):
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        self.model = model
        self.savedir = savedir

    def to_savedmodel(self, savename: str, version: int = 1):
        savepath = os.path.join(self.savedir, savename, str(version))

        # Saving model as SavedModel format
        tf.saved_model.save(self.model, savepath)

        # Printing the output nodes name (useful for converting into TensorFlowJS format
        output_nodes_name = self.model.output_names[0]
        print(f"Output node names: {output_nodes_name}")

    def to_tfjs(self, savename: str):
        savepath = os.path.join(self.savedir, savename)

        # Saving model as tfjs format
        tfjs.converters.save_keras_model(self.model, savepath)


if __name__ == "__main__":
    params = "./pretrain/classifier/classifier.hdf5"
    savedir = os.path.dirname(params)
    input_shape = (224, 224)

    model = generate_model(params, shape=input_shape)
    saving = SaveModel(model, savedir=savedir)  # Instantiate the saving object

    # Saving to other format
    savename = "mobilenet-spoof"
    saving.to_tfjs(savename)  # save model to TFJS format

    print(f"Finished saving on {savedir}!")

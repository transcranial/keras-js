import sys
import os
import h5py
import numpy as np
import json
import argparse
import uuid
import model_pb2


class Encoder(object):
    """Encoder class.

    Takes as input a Keras model saved in hdf5 format that includes the model architecture with the weights.
    This is the resulting file from running the command:

    ```
    model.save('my_model.h5')
    ```

    See https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state
    """

    def __init__(self, hdf5_model_filepath, name):
        if not hdf5_model_filepath:
            raise Exception('hdf5_model_filepath must be provided.')
        self.hdf5_model_filepath = hdf5_model_filepath
        self.name = name

        self.create_model()

    def create_model(self):
        """Initializes a model from the protobuf definition.
        """
        self.model = model_pb2.Model()
        self.model.id = str(uuid.uuid4())
        self.model.name = self.name

    def serialize(self):
        """serialize method.
        Strategy for extracting the weights is adapted from the
        load_weights_from_hdf5_group method of the Container class:
        see https://github.com/fchollet/keras/blob/master/keras/engine/topology.py#L2505-L2585
        """
        hdf5_file = h5py.File(self.hdf5_model_filepath, mode='r')

        self.model.keras_version = hdf5_file.attrs['keras_version']
        self.model.backend = hdf5_file.attrs['backend']
        self.model.model_config = hdf5_file.attrs['model_config']

        f = hdf5_file['model_weights']
        for layer_name in f.attrs['layer_names']:
            g = f[layer_name]
            for weight_name in g.attrs['weight_names']:
                weight_value = g[weight_name].value
                w = self.model.model_weights.add()
                w.layer_name = layer_name
                w.weight_name = weight_name
                w.shape.extend(list(weight_value.shape))
                w.type = 'float32'
                w.data = weight_value.astype(np.float32).tobytes()

        hdf5_file.close()

    def save(self):
        """Saves as binary protobuf message
        """
        pb_model_filepath = os.path.join(os.path.dirname(self.hdf5_model_filepath),
                                         '{}.bin'.format(self.name))
        with open(pb_model_filepath, 'wb') as f:
            f.write(self.model.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_model_filepath')
    parser.add_argument('--name', type=str, required=False,
                        help='model name (defaults to filename without extension if not provided)')
    args = parser.parse_args()

    hdf5_model_filepath = args.hdf5_model_filepath

    if args.name is not None:
        name = args.name
    else:
        name = os.path.basename(hdf5_model_filepath).split('.')[0]

    encoder = Encoder(hdf5_model_filepath, name)
    encoder.serialize()
    encoder.save()

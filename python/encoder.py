#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import numpy as np
import argparse
import uuid
import model_pb2


def quantize_arr(arr):
    """Quantization based on linear rescaling over min/max range.
    """
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val > 0:
        quantized = np.round(255 * (arr - min_val) / (max_val - min_val))
    else:
        quantized = np.zeros(arr.shape)
    quantized = quantized.astype(np.uint8)
    min_val = min_val.astype(np.float32)
    max_val = max_val.astype(np.float32)
    return quantized, min_val, max_val


class Encoder:
    """Encoder class.

    Takes as input a Keras model saved in hdf5 format that includes the model architecture with the weights.
    This is the resulting file from running the command:

    ```
    model.save('my_model.h5')
    ```

    See https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state
    """

    def __init__(self, hdf5_model_filepath, name, quantize):
        if not hdf5_model_filepath:
            raise Exception('hdf5_model_filepath must be provided.')
        self.hdf5_model_filepath = hdf5_model_filepath
        self.name = name
        self.quantize = quantize

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
        see https://github.com/keras-team/keras/blob/master/keras/engine/topology.py#L2505-L2585
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
                if self.quantize:
                    w.type = 'uint8'
                    quantized, min_val, max_val = quantize_arr(weight_value)
                    w.data = quantized.astype(np.uint8).tobytes()
                    w.quantize_min = min_val
                    w.quantize_max = max_val
                else:
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
        print('Saved to binary file {}'.format(os.path.abspath(pb_model_filepath)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_model_filepath')
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='model name (defaults to filename without extension if not provided)')
    parser.add_argument('-q', '--quantize', action='store_true', required=False,
                        help='quantize weights to 8-bit unsigned int')
    args = parser.parse_args()

    hdf5_model_filepath = args.hdf5_model_filepath

    if args.name is not None:
        name = args.name
    else:
        name = os.path.splitext(os.path.basename(hdf5_model_filepath))[0]

    quantize = args.quantize

    encoder = Encoder(hdf5_model_filepath, name, quantize)
    encoder.serialize()
    encoder.save()

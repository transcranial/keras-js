# [Keras.js](https://transcranial.github.io/keras-js)

Run trained [Keras](https://github.com/fchollet/keras) models in your browser, GPU-powered using WebGL. Models are serialized directly from the Keras JSON-format configuration file and associated HDF5 weights.

### [Interactive Demos](https://transcranial.github.io/keras-js)

- Basic Convnet for MNIST

- Convolutional Variational Autoencoder, trained on MNIST

- 50-layer Residual Network, trained on ImageNet

- Inception V3, trained on ImageNet

- Bidirectional LSTM for IMDB sentiment classification

<p align="center">
  <img src="demos/assets/mnist-cnn.png" height="120" width="auto" />
  <img src="demos/assets/resnet50.png" height="120" width="auto" />
  <img src="demos/assets/inception-v3.png" height="120" width="auto" />
  <img src="demos/assets/imdb-bidirectional-lstm.png" height="120" width="auto" />
</p>

### Why?

- Eliminate need for backend infrastructure or API calls

- Offload computation entirely to client browsers

- Interactive apps

### Usage

See `demos/src/` for source code of real examples.

1. Works for both `Model` and `Sequential`:

  ```py
  model = Sequential()
  model.add(...)
  ...
  ```

  ```py
  ...
  model = Model(input=..., output=...)
  ```

  Once trained, save the weights and export model architecture config:

  ```py
  model.save_weights('model.hdf5')
  with open('model.json', 'w') as f:
    f.write(model.to_json())
  ```

  See jupyter notebooks of demos for details: `demos/notebooks/`.

2. Run the encoder script on the HDF5 weights file:

  ```sh
  $ python encoder.py /path/to/model.hdf5
  ```

  This will produce 2 files in the same folder as the HDF5 weights: `model_weights.buf` and `model_metadata.json`.

3. The 3 files required for Keras.js are:

  - the model file: `model.json`

  - the weights file: `model_weights.buf`

  - the weights metadata file: `model_metadata.json`

4. GPU support is powered by [weblas](https://github.com/waylonflinn/weblas). Include the Keras.js and Weblas libraries:

  ```html
  <script src="lib/weblas.js"></script>
  <script src="dist/keras.js"></script>
  ```

5. Create new model

  On instantiation, data is loaded over XHR (same-domain or CORS required), and layers are initialized as directed acyclic graph. Class method `ready()` returns a Promise which resolves when these steps are complete. Then, use `predict()` to run data through the model, which also returns a Promise.

  ```js
  const model = new KerasJS.Model({
    filepaths: {
      model: 'url/path/to/model.json',
      weights: 'url/path/to/model_weights.buf',
      metadata: 'url/path/to/model_metadata.json'
    }
    gpu: true
  })

  model.ready().then(() => {

    // input data object keyed by names of the input layers
    // or `input` for Sequential models
    // values are the flattened Float32Array data
    // (input tensor shapes are specified in the model config)
    const inputData = {
      'input_1': new Float32Array(data)
    }

    // make predictions
    // outputData is an object keyed by names of the output layers
    // or `output` for Sequential models
    model.predict(inputData).then(outputData => {
      // e.g.,
      // outputData['fc1000']
    })
  })
  ```

### Available layers

  - advanced activations: `LeakyReLU`, `PReLU`, `ELU`, `ParametricSoftplus`, `ThresholdedReLU`, `SReLU`

  - convolutional: `Convolution1D`, `Convolution2D`, `AtrousConvolution2D`, `SeparableConvolution2D`, `Deconvolution2D`, `Convolution3D`, `UpSampling1D`, `UpSampling2D`, `UpSampling3D`, `ZeroPadding1D`, `ZeroPadding2D`, `ZeroPadding3D`

  - core: `Dense`, `Activation`, `Dropout`, `SpatialDropout2D`, `SpatialDropout3D`, `Flatten`, `Reshape`, `Permute`, `RepeatVector`, `Merge`, `Highway`, `MaxoutDense`

  - embeddings: `Embedding`

  - normalization: `BatchNormalization`

  - pooling: `MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`, `AveragePooling1D`, `AveragePooling2D`, `AveragePooling3D`, `GlobalMaxPooling1D`, `GlobalAveragePooling1D`, `GlobalMaxPooling2D`, `GlobalAveragePooling2D`

  - recurrent: `SimpleRNN`, `LSTM`, `GRU`

  - wrappers: `Bidirectional`, `TimeDistributed`

  **Layers not yet implemented**

  Lambda cannot be implemented directly at this point, but will eventually create a mechanism for defining computational logic through JavaScript.

  - core: `Lambda`

  - convolutional: `Cropping1D`, `Cropping2D`, `Cropping3D`

  - locally-connected: `LocallyConnected1D`, `LocallyConnected2D`

  - noise: `GaussianNoise`, `GaussianDropout`

### Notes

**WebWorkers and their limitations**

Kera.js can be run in a WebWorker separate from the main thread. Because Keras.js performs a lot of synchronous computations, this can prevent the UI from being affected. However, one of the biggest limitations of WebWorkers is the lack of `<canvas>` (and thus WebGL) access. So the benefits gained by running Keras.js in a separate thread are offset by the necessity of running it in CPU-mode only. In other words, one can run Keras.js in GPU mode only on the main thread.

**WebGL MAX_TEXTURE_SIZE**

In GPU mode, tensor objects are encoded as WebGL textures prior to computations. The size of these tensors are limited by `gl.getParameter(gl.MAX_TEXTURE_SIZE)`, which differs by hardware/platform. See [here](http://webglstats.com/) for typical expected values. The may be an issue in convolution layers after `im2col`. For example, in the Inception V3 network demo, `im2col` in the 1st convolutional layer creates a 22201 x 27 matrix, and 21609 x 288 matrices in the 2nd and 3rd convolutional layers. The size along the first dimension exceeds most `MAX_TEXTURE_SIZE`, 16384, and therefore must be split. Matrix mutiplications are performed with the weights for each split tensor and then combined. In this case, a `weblasTensorsSplit` property is available on the `Tensor` object when `createWeblasTensor()` is called (see `src/Tensor.js`). See `src/layers/convolutional/Convolution2D.js` for an example of its usage.

### Development / Testing

There are extensive tests for each implemented layer. See `notebooks/` for jupyter notebooks generating the data for all these tests.

```sh
$ npm install
```

To run all tests run `npm run server` and simply go to [http://localhost:3000/test/](http://localhost:3000/test/). All tests will automatically run. Open up your browser devtools for additional test data info.

For development, run:

```sh
$ npm run watch
```

Editing of any file in `src/` will trigger webpack to update `dist/keras.js`.

To create a production UMD webpack build, output to `dist/keras.js`, run:

```sh
$ npm run build
```

### License

[MIT](https://github.com/transcranial/keras-js/blob/master/LICENSE)

# [Keras.js](https://transcranial.github.io/keras-js)

Run trained [Keras](https://github.com/fchollet/keras) models in your browser, GPU-powered using WebGL.

### [Interactive Demos](https://transcranial.github.io/keras-js)

- Basic convolutional neural network for MNIST

- Convolutional variational autoencoder trained on MNIST, with image generation from the latent space (decoder part)

- 50-layer residual network, trained on ImageNet

- Inception V3 network, trained on ImageNet

- Bidirectional LSTM for sentiment classification, trained on the IMDB movie reviews dataset

### Why?

- Eliminate need for backend infrastructure.

- Interactivity

- Visualization

- Education (great example being [convnetjs](https://github.com/karpathy/convnetjs), of course)

- Debugging of your neural network

### Usage

1.

  ```py
  model = Sequential()
  ...
  ```

  ```py
  model = Model()
  ```

2.

3.

4. GPU support is powered by [weblas](https://github.com/waylonflinn/weblas).

### API

### Notes

**WebWorkers and their limitations**

Kera.js can be run in a WebWorker separate from the main thread. Because Keras.js performs a lot of synchronous computations, this can prevent the UI from being affected. However, one of the biggest limitations of WebWorkers is the lack of `<canvas>` (and thus WebGL) access. So the benefits gained by running Keras.js in a separate thread are offset by the necessity of running it in CPU-mode only. In other words, one can run Keras.js in GPU mode only on the main thread.

**WebGL MAX_TEXTURE_SIZE**

In GPU mode, tensor objects are encoded as WebGL textures prior to computations. The size of these tensors are limited by `gl.getParameter(gl.MAX_TEXTURE_SIZE)`, which differs by hardware/platform. See [here](http://webglstats.com/) for typical expected values. The may be an issue in convolution layers after `im2col`. For example, in the Inception V3 network demo, `im2col` in the 1st convolutional layer creates a 22201 x 27 matrix, and 21609 x 288 matrices in the 2nd and 3rd convolutional layers. The size along the first dimension exceeds most `MAX_TEXTURE_SIZE`, 16384, and therefore must be split. Matrix mutiplications are performed with the weights for each split tensor and then combined. In this case, a `weblasTensorsSplit` property is available on the `Tensor` object when `createWeblasTensor()` is called (see `src/Tensor.js`). See `src/layers/convolutional/Convolution2D.js` for an example of its usage.

### Development / Testing

### License

[MIT](https://github.com/transcranial/keras-js/blob/master/LICENSE)

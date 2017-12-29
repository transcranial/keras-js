<p align="center">
  <a href="https://transcranial.github.io/keras-js">
    <img src="https://cdn.rawgit.com/transcranial/keras-js/73aa4cca/assets/logo.svg" width="300px" />
  </a>
</p>

<p align="center">
  <strong>Run Keras models in the browser, with GPU support using WebGL</strong>
</p>

<div align="center">
  <h3>
    <a href="https://transcranial.github.io/keras-js">Interactive Demos</a>
    <span> | </span>
    <a href="https://transcranial.github.io/keras-js-docs">Documentation</a>
  </h3>
</div>

<p align="center">
  <a href="https://cdnjs.com/libraries/keras-js">
    <img src="https://img.shields.io/cdnjs/v/keras-js.svg?style=flat-square" />
  </a>
  <a href="https://www.npmjs.com/package/keras-js">
    <img src="https://img.shields.io/npm/v/keras-js.svg?style=flat-square" />
  </a>
</p>

<br/>

---

Run [Keras](https://github.com/keras-team/keras) models in the browser, with GPU support provided by WebGL 2. Models can be run in Node.js as well, but only in CPU mode. Because Keras abstracts away a number of frameworks as backends, the models can be trained in any backend, including TensorFlow, CNTK, etc.

Library version compatibility: Keras 2.1.2

## [Interactive Demos](https://transcranial.github.io/keras-js)

<p align="center">
  <a href="https://transcranial.github.io/keras-js"><img src="demos/assets/mnist-cnn.png" height="120" width="auto" /></a>
  <a href="https://transcranial.github.io/keras-js"><img src="demos/assets/resnet50.png" height="120" width="auto" /></a>
  <a href="https://transcranial.github.io/keras-js"><img src="demos/assets/inception-v3.png" height="120" width="auto" /></a>
  <a href="https://transcranial.github.io/keras-js"><img src="demos/assets/imdb-bidirectional-lstm.png" height="120" width="auto" /></a>
</p>

Check out the `demos/` directory for real examples running Keras.js in VueJS.

* Basic Convnet for MNIST
* Convolutional Variational Autoencoder, trained on MNIST
* Auxiliary Classifier Generative Adversarial Networks (AC-GAN) on MNIST
* 50-layer Residual Network, trained on ImageNet
* Inception v3, trained on ImageNet
* DenseNet-121, trained on ImageNet
* SqueezeNet v1.1, trained on ImageNet
* Bidirectional LSTM for IMDB sentiment classification

## [Documentation](https://transcranial.github.io/keras-js-docs)

[MIT License](https://github.com/transcranial/keras-js/blob/master/LICENSE)

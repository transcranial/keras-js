/* global Vue */
import './home.css'

const DEMO_INFO_DEV = [
  {
    title: 'Basic Convnet for MNIST',
    imagePath: '/demos/assets/mnist-cnn.png'
  },
  {
    title: 'Convolutional Variational Autoencoder, trained on MNIST',
    imagePath: '/demos/assets/mnist-vae.png'
  },
  {
    title: '50-layer Residual Network, trained on ImageNet',
    imagePath: '/demos/assets/resnet50.png'
  },
  {
    title: 'Inception V3, trained on ImageNet',
    imagePath: '/demos/assets/inception-v3.png'
  },
  {
    title: 'Bidirectional LSTM for IMDB sentiment classification',
    imagePath: '/demos/assets/imdb-bidirectional-lstm.png'
  }
]

const DEMO_INFO_PROD = [
  {
    title: 'Basic Convnet for MNIST',
    path: 'mnist-cnn',
    imagePath: 'demos/assets/mnist-cnn.png'
  },
  {
    title: 'Convolutional Variational Autoencoder, trained on MNIST',
    path: 'mnist-vae',
    imagePath: 'demos/assets/mnist-vae.png'
  },
  {
    title: '50-layer Residual Network, trained on ImageNet',
    path: 'resnet50',
    imagePath: 'demos/assets/resnet50.png'
  },
  {
    title: 'Inception V3, trained on ImageNet',
    path: 'inception-v3',
    imagePath: 'demos/assets/inception-v3.png'
  },
  {
    title: 'Bidirectional LSTM for IMDB sentiment classification',
    path: 'imdb-bidirectional-lstm',
    imagePath: 'demos/assets/imdb-bidirectional-lstm.png'
  }
]

const DEMO_INFO = (process.env.NODE_ENV === 'production') ? DEMO_INFO_PROD : DEMO_INFO_DEV

export const Home = Vue.extend({
  template: require('raw!./home.template.html'),

  data: function () {
    return {
      demoInfo: DEMO_INFO
    }
  }
})

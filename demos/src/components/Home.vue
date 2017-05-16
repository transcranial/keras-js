<template>
  <div class="demo home">
    <div class="title"><span></span></div>
    <a v-for="info in demoInfo" :href="`#/${info.path}`">
      <div class="demo-card">
        <div class="demo-card-image"><img :src="info.imagePath" /></div>
        <div class="demo-card-heading">{{ info.title }}</div>
      </div>
    </a>
  </div>
</template>

<script>
const DEMO_INFO_DEV = [
  { title: 'Basic Convnet for MNIST', path: 'mnist-cnn', imagePath: '/demos/assets/mnist-cnn.png' },
  {
    title: 'Convolutional Variational Autoencoder, trained on MNIST',
    path: 'mnist-vae',
    imagePath: '/demos/assets/mnist-vae.png'
  },
  {
    title: '50-layer Residual Network, trained on ImageNet',
    path: 'resnet50',
    imagePath: '/demos/assets/resnet50.png'
  },
  { title: 'Inception V3, trained on ImageNet', path: 'inception-v3', imagePath: '/demos/assets/inception-v3.png' },
  {
    title: 'Bidirectional LSTM for IMDB sentiment classification',
    path: 'imdb-bidirectional-lstm',
    imagePath: '/demos/assets/imdb-bidirectional-lstm.png'
  }
]

const DEMO_INFO_PROD = [
  { title: 'Basic Convnet for MNIST', path: 'mnist-cnn', imagePath: 'demos/assets/mnist-cnn.png' },
  {
    title: 'Convolutional Variational Autoencoder, trained on MNIST',
    path: 'mnist-vae',
    imagePath: 'demos/assets/mnist-vae.png'
  },
  { title: '50-layer Residual Network, trained on ImageNet', path: 'resnet50', imagePath: 'demos/assets/resnet50.png' },
  { title: 'Inception V3, trained on ImageNet', path: 'inception-v3', imagePath: 'demos/assets/inception-v3.png' },
  {
    title: 'Bidirectional LSTM for IMDB sentiment classification',
    path: 'imdb-bidirectional-lstm',
    imagePath: 'demos/assets/imdb-bidirectional-lstm.png'
  }
]

const DEMO_INFO = process.env.NODE_ENV === 'production' ? DEMO_INFO_PROD : DEMO_INFO_DEV

export default {
  data: function() {
    return { demoInfo: DEMO_INFO }
  }
}
</script>

<style scoped>
@import '../variables.css';

.demo-card {
  width: 100%;
  max-width: 1000px;
  margin: 30px auto;
  background: white;
  border: 2px solid white;
  border-radius: 10px;
  transition: border-color 0.2s ease-in;
  cursor: default;
  user-select: none;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  overflow: hidden;

  &:hover {
    border-color: var(--color-1);
    cursor: pointer;

    & .demo-card-heading {
      color: var(--color-1);
    }
  }
}

.demo-card-heading {
  flex: 1;
  padding: 20px;
  text-align: center;
  color: var(--color-2);
  font-family: var(--font-1);
  font-size: 20px;
  transition: color 0.2s ease-in;
}

.demo-card-image {
  & img {
    width: 180px;
    height: auto;
  }
}
</style>

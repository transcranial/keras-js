<template>
  <div class="demo home text-xs-center">
    <div v-for="info in demoInfo" :key="info.path" style="width:100%">
      <a :href="`#/${info.path}`">
        <div class="demo-card">
          <div class="demo-card-image"><img :src="info.imagePath" /></div>
          <div class="demo-card-heading">{{ info.title }}</div>
        </div>
      </a>
    </div>
  </div>
</template>

<script>
const DEMO_INFO = [
  { title: 'Basic Convnet for MNIST', path: 'mnist-cnn', imagePath: 'demos/assets/mnist-cnn.png' },
  {
    title: 'Convolutional Variational Autoencoder, trained on MNIST',
    path: 'mnist-vae',
    imagePath: 'demos/assets/mnist-vae.png'
  },
  {
    title: 'Auxiliary Classifier Generative Adversarial Network, trained on MNIST',
    path: 'mnist-acgan',
    imagePath: 'demos/assets/mnist-acgan.png'
  },
  { title: '50-layer Residual Network, trained on ImageNet', path: 'resnet50', imagePath: 'demos/assets/resnet50.png' },
  { title: 'Inception v3, trained on ImageNet', path: 'inception-v3', imagePath: 'demos/assets/inception-v3.png' },
  { title: 'DenseNet-121, trained on ImageNet', path: 'densenet121', imagePath: 'demos/assets/densenet121.png' },
  {
    title: 'SqueezeNet v1.1, trained on ImageNet',
    path: 'squeezenet-v1.1',
    imagePath: 'demos/assets/squeezenet-v1.1.png'
  },
  {
    title: 'Bidirectional LSTM for IMDB sentiment classification',
    path: 'imdb-bidirectional-lstm',
    imagePath: 'demos/assets/imdb-bidirectional-lstm.png'
  },
  {
    title: 'Image Super-Resolution CNNs',
    path: 'image-super-resolution',
    imagePath: 'demos/assets/image-super-resolution.png'
  }
]

export default {
  data: function() {
    return { demoInfo: DEMO_INFO }
  }
}
</script>

<style scoped lang="postcss">
@import '../variables.css';

.demo-card {
  font-family: var(--font-sans-serif);
  width: 100%;
  max-width: 1000px;
  height: 90px;
  margin: 30px auto;
  background: white;
  border: 1px solid whitesmoke;
  cursor: default;
  user-select: none;
  display: inline-flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  overflow: hidden;
  box-shadow: 3px 3px #cccccc;
  transition: box-shadow 0.2s ease-out;

  &:first-child {
    margin-top: 0;
  }

  &:hover {
    box-shadow: 3px 3px 5px var(--color-green-light);
    cursor: pointer;

    & .demo-card-heading {
      color: var(--color-green);
    }
  }
}

.demo-card-heading {
  flex: 1;
  padding: 20px;
  text-align: center;
  color: var(--color-lightgray);
  font-size: 18px;
  transition: color 0.2s ease-out;
}

.demo-card-image {
  height: 90px;

  & img {
    width: auto;
    height: 100%;
  }
}
</style>

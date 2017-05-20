<template>
  <div class="info-panel">
    <div class="info-panel-text" v-if="currentView === 'mnist-cnn'">
      Interactive demo of a simple convnet trained on MNIST (see <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/demos/notebooks/mnist_cnn.ipynb">Jupyter notebook</a>). All computation performed entirely in your browser. Toggling GPU on/off shouldn't reveal any significant speed differences, as this is a fairly small network. In the architecture diagram below, intermediate outputs at each layer are also visualized.
    </div>
    <div class="info-panel-text" v-else-if="currentView === 'mnist-vae'">
      Modified from the MNIST convolution/deconvolution variational autoencoder example <a target="_blank" href="https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder_deconv.py">here</a>. The network demonstrated here is the generative decoder portion (see <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/demos/notebooks/mnist_vae.ipynb">Jupyter notebook</a>). The network generates an image through a series of Conv2DTranspose layers from coordinates in the 2D latent space. All computation performed entirely in your browser. Toggling GPU on/off shouldn't reveal any significant speed differences, as this is a fairly small network. In the architecture diagram below, intermediate outputs at each layer are also visualized.
    </div>
    <div class="info-panel-text" v-else-if="currentView === 'mnist-acgan'">
      Interactive demo of MNIST digit generation, using the generator network trained in an auxillary classifier generative adversarial network (AC-GAN). See <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/demos/notebooks/mnist_acgan.ipynb">Jupyter notebook</a>. Modified based on original GitHub repo <a target="_blank" href="https://github.com/lukedeo/keras-acgan">lukedeo/keras-acgan</a>. During training the discriminator must both tease apart real images from fake synthetic images, as well as perform classification on the images. Here, during generation, we can condition upon the digit class.
    </div>
    <div class="info-panel-text" v-else-if="currentView === 'resnet50'">
      Note that ~100 MB of weights must be loaded. We use the Keras architecture from <a target="_blank" href="https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py">here</a> and pretrained weights from <a target="_blank" href="https://github.com/fchollet/deep-learning-models">here</a>. Enter any valid image URL as input to the network. You can also select from a list of prepopulated image URLs. The endpoint must have CORS enabled, to enable us to extract the numeric data from the canvas element, so not all URLs will work. Imgur and <a target="_blank" href="https://www.flickr.com/search/?text=&license=2%2C3%2C4%2C5%2C6%2C9&sort=interestingness-desc">Flickr creative commons</a> all work, and are good places to start. After running the network, the top-5 classes are displayed. Keep in mind also we are limited to the <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/demos/src/utils/imagenet.js">1,000 classes of ImageNet</a>. Keep in mind that this is image classification and not object detection, so the network is forced to output a single class through softmax. Best results are on images where the classification target spans a large portion of the image. All computation performed entirely in your browser. Toggling GPU on should offer significant speedups compared to CPU. Running the network may still take several seconds (optimizations to come). With "show computational flow" toggled, computation through the network will be shown in the architecture diagram (scroll down as computation is performed layer by layer).
    </div>
    <div class="info-panel-text" v-else-if="currentView === 'inception-v3'">
      Note that ~100 MB of weights must be loaded. We use the Keras architecture from <a target="_blank" href="https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py">here</a> and pretrained weights from <a target="_blank" href="https://github.com/fchollet/deep-learning-models">here</a>. Enter any valid image URL as input to the network. You can also select from a list of prepopulated image URLs. The endpoint must have CORS enabled, to enable us to extract the numeric data from the canvas element, so not all URLs will work. Imgur and <a target="_blank" href="https://www.flickr.com/search/?text=&license=2%2C3%2C4%2C5%2C6%2C9&sort=interestingness-desc">Flickr creative commons</a> all work, and are good places to start. After running the network, the top-5 classes are displayed. Keep in mind also we are limited to the <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/demos/src/utils/imagenet.js">1,000 classes of ImageNet</a>. Keep in mind that this is image classification and not object detection, so the network is forced to output a single class through softmax. Best results are on images where the classification target spans a large portion of the image. All computation performed entirely in your browser. Toggling GPU on should offer significant speedups compared to CPU. Running the network may still take several seconds (optimizations to come). With "show computational flow" toggled, computation through the network will be shown in the architecture diagram (scroll down as computation is performed layer by layer).
    </div>
    <div class="info-panel-text" v-else-if="currentView === 'squeezenet-v1.1'">
      In contrast to ResNet-50 and Inception-V3, the size of the weights for SqueezeNet is relatively minimal, less than 5 MB. We use the architecture and pretrained weights from <a target="_blank" href="https://github.com/rcmalli/keras-squeezenet">rcmalli/keras-squeezenet</a> (original at <a target="_blank" href="https://github.com/DeepScale/SqueezeNet">DeepScale/SqueezeNet</a>). See Jupyter notebook <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/notebooks/demos/squeezenet_v1.1.ipynb">here</a>. Enter any valid image URL as input to the network. You can also select from a list of prepopulated image URLs. The endpoint must have CORS enabled, to enable us to extract the numeric data from the canvas element, so not all URLs will work. Imgur and <a target="_blank" href="https://www.flickr.com/search/?text=&license=2%2C3%2C4%2C5%2C6%2C9&sort=interestingness-desc">Flickr creative commons</a> all work, and are good places to start. After running the network, the top-5 classes are displayed. Keep in mind also we are limited to the <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/demos/src/utils/imagenet.js">1,000 classes of ImageNet</a>. Keep in mind that this is image classification and not object detection, so the network is forced to output a single class through softmax. Best results are on images where the classification target spans a large portion of the image. All computation performed entirely in your browser.
    </div>
    <div class="info-panel-text" v-else-if="currentView === 'imdb-bidirectional-lstm'">
      This demo is modified from the Keras <a target="_blank" href="https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py">example</a> demonstrating the Bidirectional wrapper class around an LSTM layer. Click on "load sample text" to populate the textbox with a sample IMDB movie review (preprocessed) from the test set (not used during training). You can also enter your own text into the textbox, but keep in mind the model was trained on IMDB movie reviews only (see the corresponding <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/demos/notebooks/imdb_bidirectional_lstm.ipynb">Jupyter notebook</a>). The result is a number from 0 (negative) to 1 (positive). We visualize the contributions from each word by running the forward+backward concatenated hidden state corresponding to each word through the final Dense layer.
    </div>
  </div>
</template>

<script>
export default {
  props: {
    currentView: { type: String, default: 'home' }
  }
}
</script>

<style scoped>
@import '../variables.css';

.info-panel {
  padding: 20px 10px;
  margin: 20px;

  & .info-panel-text {
    width: 100%;
    color: var(--color-darkgray);
    font-size: 14px;
    text-align: justify;

    & a {
      color: var(--color-green);
      transition: color 0.2s ease-in;

      &:hover {
        color: var(--color-green-light);
      }
    }
  }
}
</style>

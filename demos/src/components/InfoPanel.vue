<template>
  <v-dialog v-model="show" max-width="600px">
    <v-card>
      <v-card-title><div class="info-panel-title">More Information</div></v-card-title>
      <v-card-text>
        <div class="info-panel-text" v-if="currentView === 'mnist-cnn'">
          <p>Interactive demo of a simple convnet trained on MNIST (see <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/notebooks/demos/mnist_cnn.ipynb">Jupyter notebook</a>).</p><p>All computation performed entirely in your browser. Toggling GPU on/off shouldn't reveal any significant speed differences, as this is a fairly small network. In the architecture diagram below, intermediate outputs at each layer are also visualized.</p>
        </div>
        <div class="info-panel-text" v-else-if="currentView === 'mnist-vae'">
          <p>Modified from the MNIST convolution/deconvolution variational autoencoder example <a target="_blank" rel="noopener noreferrer" href="https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py">here</a>. The network demonstrated here is the generative decoder portion (see <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/notebooks/demos/mnist_vae.ipynb">Jupyter notebook</a>).</p><p>The network generates an image through a series of Conv2DTranspose layers from coordinates in the 2D latent space. All computation performed entirely in your browser. Toggling GPU on/off shouldn't reveal any significant speed differences, as this is a fairly small network. In the architecture diagram below, intermediate outputs at each layer are also visualized.</p>
        </div>
        <div class="info-panel-text" v-else-if="currentView === 'mnist-acgan'">
          <p>Interactive demo of MNIST digit generation, using the generator network trained in an auxillary classifier generative adversarial network (AC-GAN). See <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/notebooks/demos/mnist_acgan.ipynb">Jupyter notebook</a>. Modified based on original GitHub repo <a target="_blank" rel="noopener noreferrer" href="https://github.com/lukedeo/keras-acgan">lukedeo/keras-acgan</a>.</p><p>During training the discriminator must both tease apart real images from fake synthetic images, as well as perform classification on the images. Here, during generation, we can condition upon the digit class.</p>
        </div>
        <div class="info-panel-text" v-else-if="currentView === 'resnet50'">
          <p>Note that ~25 MB of weights must be loaded (8-bit quantization is used). We use the Keras architecture from <a target="_blank" rel="noopener noreferrer" href="https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py">here</a> and pretrained weights from <a target="_blank" rel="noopener noreferrer" href="https://github.com/keras-team/deep-learning-models">here</a>.</p><p>Enter any valid image URL as input to the network. You can also select from a list of prepopulated image URLs. The endpoint must have CORS enabled, to enable us to extract the numeric data from the canvas element, so not all URLs will work. Imgur and <a target="_blank" rel="noopener noreferrer" href="https://www.flickr.com/search/?text=&license=2%2C3%2C4%2C5%2C6%2C9&sort=interestingness-desc">Flickr creative commons</a> all work, and are good places to start. After running the network, the top-5 classes are displayed. Keep in mind also we are limited to the <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/demos/src/utils/imagenet.js">1,000 classes of ImageNet</a>.</p><p>All computation performed entirely in your browser. Toggling GPU on should offer significant speedups compared to CPU.</p>
        </div>
        <div class="info-panel-text" v-else-if="currentView === 'inception-v3'">
          <p>Note that ~25 MB of weights must be loaded (8-bit quantization is used). We use the Keras architecture from <a target="_blank" rel="noopener noreferrer" href="https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py">here</a> and pretrained weights from <a target="_blank" rel="noopener noreferrer" href="https://github.com/keras-team/deep-learning-models">here</a>.</p><p>Enter any valid image URL as input to the network. You can also select from a list of prepopulated image URLs. The endpoint must have CORS enabled, to enable us to extract the numeric data from the canvas element, so not all URLs will work. Imgur and <a target="_blank" rel="noopener noreferrer" href="https://www.flickr.com/search/?text=&license=2%2C3%2C4%2C5%2C6%2C9&sort=interestingness-desc">Flickr creative commons</a> all work, and are good places to start. After running the network, the top-5 classes are displayed. Keep in mind also we are limited to the <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/demos/src/utils/imagenet.js">1,000 classes of ImageNet</a>.</p><p>All computation performed entirely in your browser. Toggling GPU on should offer significant speedups compared to CPU.</p>
        </div>
        <div class="info-panel-text" v-else-if="currentView === 'squeezenet-v1.1'">
          <p>In contrast to ResNet-50 and Inception-V3, the size of the weights file for SqueezeNet is tiny at 1.3 MB (using 8-bit quantization). We use the architecture and pretrained weights from <a target="_blank" rel="noopener noreferrer" href="https://github.com/rcmalli/keras-squeezenet">rcmalli/keras-squeezenet</a> (original at <a target="_blank" rel="noopener noreferrer" href="https://github.com/DeepScale/SqueezeNet">DeepScale/SqueezeNet</a>). See Jupyter notebook <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/notebooks/demos/squeezenet_v1.1.ipynb">here</a>.</p><p>Enter any valid image URL as input to the network. You can also select from a list of prepopulated image URLs. The endpoint must have CORS enabled, to enable us to extract the numeric data from the canvas element, so not all URLs will work. Imgur and <a target="_blank" rel="noopener noreferrer" href="https://www.flickr.com/search/?text=&license=2%2C3%2C4%2C5%2C6%2C9&sort=interestingness-desc">Flickr creative commons</a> all work, and are good places to start. After running the network, the top-5 classes are displayed. Keep in mind also we are limited to the <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/demos/src/utils/imagenet.js">1,000 classes of ImageNet</a>.</p><p>All computation performed entirely in your browser. Toggling GPU on should offer significant speedups compared to CPU.</p>
        </div>
        <div class="info-panel-text" v-else-if="currentView === 'imdb-bidirectional-lstm'">
          <p>This demo is modified from the Keras <a target="_blank" rel="noopener noreferrer" href="https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py">example</a> demonstrating the Bidirectional wrapper class around an LSTM layer. Click on "load sample text" to populate the textbox with a sample IMDB movie review (preprocessed) from the test set (not used during training). You can also enter your own text into the textbox, but keep in mind the model was trained on IMDB movie reviews only (see the corresponding <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/notebooks/demos/imdb_bidirectional_lstm.ipynb">Jupyter notebook</a>).</p><p>The result is a number from 0 (negative) to 1 (positive). We visualize the contributions from each word by running the forward+backward concatenated hidden state corresponding to each word through the final Dense layer.</p>
        </div>
        <div class="info-panel-text" v-else-if="currentView === 'image-super-resolution'">
          <p>This demonstrates several CNN models for super-resolution. They are adapted from GitHub repo <a target="_blank" rel="noopener noreferrer" href="https://github.com/titu1994/Image-Super-Resolution">titu1994/Image-Super-Resolution</a>. See the corresponding <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcranial/keras-js/blob/master/notebooks/demos/image_super_resolution.ipynb">Jupyter notebook</a> preparing the models for Keras.js.</p><p>Distill ResNet SR and ResNet SR will produce sharper images while SR CNN, Expanded SR CNN, and Deep Denoising Autoencoder SR CNN will produce more subtle results. The best model may vary depending on the characteristics of the input image.</p><p>All computation performed entirely in your browser. Images are not uploaded to any servers.</p>
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn color="primary" flat @click.stop="show = false">Close</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script>
export default {
  props: {
    showInfoPanel: { type: Boolean, default: false },
    currentView: { type: String },
    close: { type: Function }
  },
  data() {
    return {
      show: false
    }
  },
  watch: {
    showInfoPanel(newVal) {
      this.show = newVal
    },
    show(newVal) {
      if (!newVal) this.close()
    }
  }
}
</script>

<style scoped lang="postcss">
@import '../variables.css';

.info-panel-title {
  margin-top: 10px;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-lightgray);
}

.info-panel-text {
  font-size: 14px;
  color: var(--color-darkgray);

  & a {
    color: var(--color-green);
    transition: color 0.2s ease-in;

    &:hover {
      color: var(--color-green-light);
    }
  }
}
</style>

import './mnist-vae.css';

import * as utils from '../utils';

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/mnist_vae/mnist_vae.json',
  weights: '/demos/data/mnist_vae/mnist_vae_weights.buf',
  metadata: '/demos/data/mnist_vae/mnist_vae_metadata.json'
};
const MODEL_FILEPATHS_PROD = {
  model: 'demos/data/mnist_vae/mnist_vae.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/mnist_vae/mnist_vae_weights.buf',
  metadata: 'demos/data/mnist_vae/mnist_vae_metadata.json'
};
const MODEL_CONFIG = {
  filepaths: process.env.NODE_ENV === 'production'
    ? MODEL_FILEPATHS_PROD
    : MODEL_FILEPATHS_DEV
};

const LAYER_DISPLAY_CONFIG = {
  dense_10: {
    heading: 'input dimensions = 2, output dimensions = 128, ReLU activation',
    scalingFactor: 2
  },
  dense_11: {
    heading: 'ReLU activation, output dimensions = 25088 (64 x 14 x 14)',
    scalingFactor: 2
  },
  reshape_4: { heading: '', scalingFactor: 2 },
  deconvolution2d_10: {
    heading: '64 3x3 filters, border mode same, 1x1 strides, ReLU activation',
    scalingFactor: 2
  },
  deconvolution2d_11: {
    heading: '64 3x3 filters, border mode same, 1x1 strides, ReLU activation',
    scalingFactor: 2
  },
  deconvolution2d_12: {
    heading: '64 2x2 filters, border mode valid, 2x2 strides, ReLU activation',
    scalingFactor: 2
  },
  convolution2d_8: {
    heading: '1 2x2 filters, border mode valid, 1x1 strides, sigmoid activation',
    scalingFactor: 2
  }
};

/**
 *
 * VUE COMPONENT
 *
 */
export const MnistVae = Vue.extend({
  props: [ 'hasWebgl' ],
  template: require('raw-loader!./mnist-vae.template.html'),
  data: function() {
    return {
      showInfoPanel: true,
      useGpu: this.hasWebgl,
      model: new KerasJS.Model(
        Object.assign({ gpu: this.hasWebgl }, MODEL_CONFIG)
      ),
      modelLoading: true,
      output: new Float32Array(27 * 27),
      crosshairsActivated: false,
      inputCoordinates: [ -0.6, -1.2 ],
      position: [ 60, 20 ],
      layerResultImages: [],
      layerDisplayConfig: LAYER_DISPLAY_CONFIG
    };
  },
  computed: {
    loadingProgress: function() {
      return this.model.getLoadingProgress();
    }
  },
  ready: function() {
    this.model.ready().then(() => {
      this.modelLoading = false;
      this.$nextTick(function() {
        this.drawPosition();
        this.getIntermediateResults();
        this.runModel();
      });
    });
  },
  methods: {
    closeInfoPanel: function() {
      this.showInfoPanel = false;
    },
    toggleGpu: function() {
      this.model.toggleGpu(!this.useGpu);
    },
    activateCrosshairs: function(e) {
      this.crosshairsActivated = true;
    },
    deactivateCrosshairs: function(e) {
      this.crosshairsActivated = false;
      this.draw(e);
    },
    draw: function(e) {
      const [ x, y ] = this.getEventCanvasCoordinates(e);
      const ctx = document.getElementById('input-canvas').getContext('2d');
      ctx.clearRect(0, 0, 200, 200);

      this.drawPosition();

      if (this.crosshairsActivated) {
        ctx.strokeStyle = '#1BBC9B';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, 200);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(200, y);
        ctx.stroke();
      }
    },
    drawPosition: function() {
      const ctx = document.getElementById('input-canvas').getContext('2d');
      ctx.clearRect(0, 0, 200, 200);
      ctx.fillStyle = '#674172';
      ctx.beginPath();
      ctx.arc(...this.position, 5, 0, Math.PI * 2, true);
      ctx.closePath();
      ctx.fill();
    },
    getEventCanvasCoordinates: function(e) {
      let { clientX, clientY } = e;
      // for touch event
      if (e.touches && e.touches.length) {
        clientX = e.touches[(0)].clientX;
        clientY = e.touches[(0)].clientY;
      }

      const canvas = document.getElementById('input-canvas');
      const { left, top } = canvas.getBoundingClientRect();
      const [ x, y ] = [ clientX - left, clientY - top ];
      return [ x, y ];
    },
    selectCoordinates: function(e) {
      const [ x, y ] = this.getEventCanvasCoordinates(e);
      if (!this.model.isRunning) {
        this.position = [ x, y ];
        this.inputCoordinates = [ x * 3 / 200 - 1.5, y * 3 / 200 - 1.5 ];
        this.draw(e);
        this.runModel();
      }
    },
    runModel: function() {
      const inputData = { input_4: new Float32Array(this.inputCoordinates) };
      this.model.predict(inputData).then(outputData => {
        this.output = outputData['convolution2d_8'];
        this.drawOutput();
        this.getIntermediateResults();
      });
    },
    drawOutput: function() {
      const ctx = document.getElementById('output-canvas').getContext('2d');
      const image = utils.image2Darray(this.output, 27, 27, [ 27, 188, 155 ]);
      ctx.putImageData(image, 0, 0);

      // scale up
      const ctxScaled = document
        .getElementById('output-canvas-scaled')
        .getContext('2d');
      ctxScaled.save();
      ctxScaled.scale(150 / 27, 150 / 27);
      ctxScaled.clearRect(
        0,
        0,
        ctxScaled.canvas.width,
        ctxScaled.canvas.height
      );
      ctxScaled.drawImage(document.getElementById('output-canvas'), 0, 0);
      ctxScaled.restore();
    },
    getIntermediateResults: function() {
      let results = [];
      for (let [ name, layer ] of this.model.modelLayersMap.entries()) {
        const layerClass = layer.layerClass || '';
        if (layerClass === 'InputLayer')
          continue;

        let images = [];
        if (layer.result && layer.result.tensor.shape.length === 3) {
          images = utils.unroll3Dtensor(layer.result.tensor);
        } else if (layer.result && layer.result.tensor.shape.length === 2) {
          images = [ utils.image2Dtensor(layer.result.tensor) ];
        } else if (layer.result && layer.result.tensor.shape.length === 1) {
          images = [ utils.image1Dtensor(layer.result.tensor) ];
        }
        results.push({ name, layerClass, images });
      }
      this.layerResultImages = results;
      setTimeout(
        () => {
          this.showIntermediateResults();
        },
        0
      );
    },
    showIntermediateResults: function() {
      this.layerResultImages.forEach((result, layerNum) => {
        const scalingFactor = this.layerDisplayConfig[result.name].scalingFactor;
        result.images.forEach((image, imageNum) => {
          const ctx = document
            .getElementById(`intermediate-result-${layerNum}-${imageNum}`)
            .getContext('2d');
          ctx.putImageData(image, 0, 0);
          const ctxScaled = document
            .getElementById(
              `intermediate-result-${layerNum}-${imageNum}-scaled`
            )
            .getContext('2d');
          ctxScaled.save();
          ctxScaled.scale(scalingFactor, scalingFactor);
          ctxScaled.clearRect(
            0,
            0,
            ctxScaled.canvas.width,
            ctxScaled.canvas.height
          );
          ctxScaled.drawImage(
            document.getElementById(
              `intermediate-result-${layerNum}-${imageNum}`
            ),
            0,
            0
          );
          ctxScaled.restore();
        });
      });
    },
    clearIntermediateResults: function() {
      this.layerResultImages.forEach((result, layerNum) => {
        const scalingFactor = this.layerDisplayConfig[result.name].scalingFactor;
        result.images.forEach((image, imageNum) => {
          const ctxScaled = document
            .getElementById(
              `intermediate-result-${layerNum}-${imageNum}-scaled`
            )
            .getContext('2d');
          ctxScaled.save();
          ctxScaled.scale(scalingFactor, scalingFactor);
          ctxScaled.clearRect(
            0,
            0,
            ctxScaled.canvas.width,
            ctxScaled.canvas.height
          );
          ctxScaled.restore();
        });
      });
    }
  }
});

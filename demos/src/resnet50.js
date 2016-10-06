/* global Vue */
import './resnet50.css'

import * as utils from './utils'

const MODEL_CONFIG = {
  filepaths: {
    model: '/demos/data/resnet50/resnet50.json',
    weights: '/demos/data/resnet50/resnet50_weights.buf',
    metadata: '/demos/data/resnet50/resnet50_metadata.json'
  },
  gpu: false
}

if (process.env.NODE_ENV === 'production') {
  Object.assign(MODEL_CONFIG, {
    filepaths: {
      model: 'demos/data/resnet50/resnet50.json',
      weights: 'https://transcranial.github.io/keras-js-demos-data/resnet50/resnet50_weights.buf',
      metadata: 'demos/data/resnet50/resnet50_metadata.json'
    }
  })
}

const LAYER_DISPLAY_CONFIG = {
}

/**
 *
 * VUE COMPONENT
 *
 */
export const ResNet50 = Vue.extend({
  template: require('raw!./resnet50.template.html'),

  data: function () {
    return {
      model: new KerasJS.Model(MODEL_CONFIG),
      modelLoading: true,
      imageURL: '',
      imageURLSelectList: [
        { name: 'cat', value: 'URL1' },
        { name: 'dog', value: 'URL2' }
      ],
      input: new Float32Array(224 * 224 * 3),
      output: new Float32Array(1000),
      layerResultImages: [],
      layerDisplayConfig: LAYER_DISPLAY_CONFIG,
      useGpu: MODEL_CONFIG.gpu
    }
  },

  computed: {
    loadingProgress: function () {
      return this.model.getLoadingProgress()
    }
  },

  created: function () {
    // initialize KerasJS model
    this.model.initialize()
    this.model.ready().then(() => {
      this.modelLoading = false
      this.getIntermediateResults()
    })
  },

  methods: {

    toggleGpu: function () {
      this.model.gpu = !this.useGpu
    },

    getIntermediateResults: function () {
      let results = []
      for (let [name, layer] of this.model.modelLayersMap.entries()) {
        const layerClass = layer.layerClass || ''
        if (layerClass === 'InputLayer') continue

        let images = []
        if (layer.result && layer.result.tensor.shape.length === 3) {
          images = utils.unroll3Dtensor(layer.result.tensor)
        } else if (layer.result && layer.result.tensor.shape.length === 2) {
          images = [utils.image2Dtensor(layer.result.tensor)]
        } else if (layer.result && layer.result.tensor.shape.length === 1) {
          images = [utils.image1Dtensor(layer.result.tensor)]
        }
        results.push({
          name,
          layerClass,
          images
        })
      }
      this.layerResultImages = results
      setTimeout(() => {
        this.showIntermediateResults()
      }, 0)
    },

    showIntermediateResults: function () {
      this.layerResultImages.forEach((result, layerNum) => {
        const scalingFactor = this.layerDisplayConfig[result.name].scalingFactor
        result.images.forEach((image, imageNum) => {
          let ctx = document.getElementById(`intermediate-result-${layerNum}-${imageNum}`).getContext('2d')
          ctx.putImageData(image, 0, 0)
          let ctxScaled = document.getElementById(`intermediate-result-${layerNum}-${imageNum}-scaled`).getContext('2d')
          ctxScaled.save()
          ctxScaled.scale(scalingFactor, scalingFactor)
          ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
          ctxScaled.drawImage(document.getElementById(`intermediate-result-${layerNum}-${imageNum}`), 0, 0)
          ctxScaled.restore()
        })
      })
    },

    clearIntermediateResults: function () {
      this.layerResultImages.forEach((result, layerNum) => {
        const scalingFactor = this.layerDisplayConfig[result.name].scalingFactor
        result.images.forEach((image, imageNum) => {
          let ctxScaled = document.getElementById(`intermediate-result-${layerNum}-${imageNum}-scaled`).getContext('2d')
          ctxScaled.save()
          ctxScaled.scale(scalingFactor, scalingFactor)
          ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
          ctxScaled.restore()
        })
      })
    }
  }
})

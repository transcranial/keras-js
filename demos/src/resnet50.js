/* global Vue, loadImage */
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
      imageURLInput: null,
      imageURLSelect: null,
      imageURLSelectList: [
        { name: 'cat', value: 'http://i.imgur.com/CzXTtJV.jpg' },
        { name: 'dog', value: 'URL2' }
      ],
      imageLoading: false,
      imageLoadingError: false,
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

  ready: function () {
    // initialize KerasJS model
    this.model.initialize()
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(function () {
        //this.getIntermediateResults()
      })
    })
  },

  methods: {

    toggleGpu: function () {
      this.model.gpu = !this.useGpu
    },

    imageURLInputChanged: function (e) {
      this.imageURLSelect = null
      this.loadImageToCanvas(this.imageURLInput)
    },

    imageURLSelectChanged: function (e) {
      this.imageURLInput = this.imageURLSelect
      this.loadImageToCanvas(this.imageURLSelect)
    },

    loadImageToCanvas: function (url) {
      this.imageLoading = true
      loadImage(
        url,
        img => {
          if (img.type === 'error') {
            this.imageLoadingError = true
            this.imageLoading = false
          } else {
            // load image data onto input canvas
            const ctx = document.getElementById('input-canvas').getContext('2d')
            ctx.drawImage(img, 0, 0)
            this.imageLoadingError = false
            this.imageLoading = false
            // model predict
            this.$nextTick(function () {
              this.runModel()
            })
          }
        },
        {
          maxWidth: 224,
          maxHeight: 224,
          cover: true,
          crop: true,
          canvas: true,
          crossOrigin: 'Anonymous'
        }
      )
    },

    runModel: function () {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

      // data processing
      // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
      let dataProcessed = new Float32Array(width * height * 3)
      for (let i = 0, len = data.length; i < len; i += 4) {
        // RGB -> BGR
        dataProcessed[i / 4 + 2] = data[i] - 103.939
        dataProcessed[i / 4 + 1] = data[i + 1] - 116.779
        dataProcessed[i / 4] = data[i + 2] - 123.68
      }

      const inputData = {
        'input_1': dataProcessed
      }
      const outputData = this.model.predict(inputData)
      this.output = outputData['fc1000']
      console.log(this.output)
      //this.getIntermediateResults()
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
          const ctx = document.getElementById(`intermediate-result-${layerNum}-${imageNum}`).getContext('2d')
          ctx.putImageData(image, 0, 0)
          const ctxScaled = document.getElementById(`intermediate-result-${layerNum}-${imageNum}-scaled`).getContext('2d')
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
          const ctxScaled = document.getElementById(`intermediate-result-${layerNum}-${imageNum}-scaled`).getContext('2d')
          ctxScaled.save()
          ctxScaled.scale(scalingFactor, scalingFactor)
          ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
          ctxScaled.restore()
        })
      })
    }
  }
})

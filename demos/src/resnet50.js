/* global Vue, loadImage */
import './resnet50.css'

import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import * as utils from './utils'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/resnet50/resnet50.json',
  weights: '/demos/data/resnet50/resnet50_weights.buf',
  metadata: '/demos/data/resnet50/resnet50_metadata.json'
}

const MODEL_FILEPATHS_PROD = {
  model: 'demos/data/resnet50/resnet50.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/resnet50/resnet50_weights.buf',
  metadata: 'demos/data/resnet50/resnet50_metadata.json'
}

const MODEL_CONFIG = {
  filepaths: (process.env.NODE_ENV === 'production') ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV
}

const IMAGE_URL_LIST = [
  { name: 'cat', value: 'http://i.imgur.com/CzXTtJV.jpg' },
  { name: 'dog', value: 'http://i.imgur.com/OB0y6MR.jpg' },
  { name: 'bridge', value: 'http://i.imgur.com/Bvke53p.jpg' }
]

const ARCHITECTURE_DIAGRAM = [
  {
    name: 'zeropadding2d_1',
    className: 'ZeroPadding2D',
    details: '3x3 padding',
    layout: 'full'
  },
]

/**
 *
 * VUE COMPONENT
 *
 */
export const ResNet50 = Vue.extend({
  props: ['hasWebgl'],

  template: require('raw!./resnet50.template.html'),

  data: function () {
    return {
      model: new KerasJS.Model(Object.assign({ gpu: this.hasWebgl }, MODEL_CONFIG)),
      modelLoading: true,
      modelRunning: false,
      imageURLInput: null,
      imageURLSelect: null,
      imageURLSelectList: IMAGE_URL_LIST,
      imageLoading: false,
      imageLoadingError: false,
      output: null,
      architectureDiagram: ARCHITECTURE_DIAGRAM,
      useGpu: this.hasWebgl
    }
  },

  computed: {
    loadingProgress: function () {
      return this.model.getLoadingProgress()
    },
    outputClasses: function () {
      if (!this.output) return []
      return utils.imagenetClassesTopK(this.output, 5)
    }
  },

  ready: function () {
    this.model.ready().then(() => {
      this.modelLoading = false
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
            this.modelRunning = true
            // model predict
            this.$nextTick(function () {
              setTimeout(() => {
                this.runModel()
              }, 200)
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
      let dataTensor = ndarray(data, [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.subseq(dataTensor.pick(null, null, 0), 103.939)
      ops.subseq(dataTensor.pick(null, null, 1), 116.779)
      ops.subseq(dataTensor.pick(null, null, 2), 123.68)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 2))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 0))

      const inputData = {
        'input_1': dataProcessedTensor.data
      }
      const outputData = this.model.predict(inputData)
      this.output = outputData['fc1000']
      this.modelRunning = false
    }
  }
})

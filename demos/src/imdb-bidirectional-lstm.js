/* global Vue */
import './imdb-bidirectional-lstm.css'

import debounce from 'lodash/debounce'
import * as utils from './utils'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.json',
  weights: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_weights.buf',
  metadata: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_weights.buf',
  metadata: 'demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_metadata.json'
}
const MODEL_CONFIG = {
  filepaths: (process.env.NODE_ENV === 'production') ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV
}

const LAYER_DISPLAY_CONFIG = {
}

/**
 *
 * VUE COMPONENT
 *
 */
export const ImdbBidirectionalLstm = Vue.extend({
  props: ['hasWebgl'],

  template: require('raw!./imdb-bidirectional-lstm.template.html'),

  data: function () {
    return {
      model: new KerasJS.Model(Object.assign({ gpu: this.hasWebgl }, MODEL_CONFIG)),
      modelLoading: true,
      input: new Float32Array(200),
      output: new Float32Array(1),
      layerResultImages: [],
      layerDisplayConfig: LAYER_DISPLAY_CONFIG,
      drawing: false,
      strokes: [],
      useGpu: this.hasWebgl
    }
  },

  computed: {
    loadingProgress: function () {
      return this.model.getLoadingProgress()
    }
  },

  ready: function () {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(function () {
        this.getIntermediateResults()
      })
    })
  },

  methods: {

    toggleGpu: function () {
      this.model.toggleGpu(!this.useGpu)
    },

    clear: function (e) {
      this.clearIntermediateResults()
      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      const ctxCenterCrop = document.getElementById('input-canvas-centercrop').getContext('2d')
      ctxCenterCrop.clearRect(0, 0, ctxCenterCrop.canvas.width, ctxCenterCrop.canvas.height)
      const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
      ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      this.output = new Float32Array(10)
      this.drawing = false
      this.strokes = []
    },

    activateDraw: function (e) {
      this.drawing = true
      this.strokes.push([])
      let points = this.strokes[this.strokes.length - 1]
      points.push(utils.getCoordinates(e))
    },

    draw: function (e) {
      if (!this.drawing) return

      const ctx = document.getElementById('input-canvas').getContext('2d')

      ctx.lineWidth = 20
      ctx.lineJoin = ctx.lineCap = 'round'
      ctx.strokeStyle = '#393E46'

      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

      let points = this.strokes[this.strokes.length - 1]
      points.push(utils.getCoordinates(e))

      // draw individual strokes
      for (let s = 0, slen = this.strokes.length; s < slen; s++) {
        points = this.strokes[s]

        let p1 = points[0]
        let p2 = points[1]
        ctx.beginPath()
        ctx.moveTo(...p1)

        // draw points in stroke
        // quadratic bezier curve
        for (let i = 1, len = points.length; i < len; i++) {
          ctx.quadraticCurveTo(...p1, ...utils.getMidpoint(p1, p2))
          p1 = points[i]
          p2 = points[i + 1]
        }
        ctx.lineTo(...p1)
        ctx.stroke()
      }
    },

    deactivateDrawAndPredict: debounce(function () {
      if (!this.drawing) return
      this.drawing = false

      const ctx = document.getElementById('input-canvas').getContext('2d')

      // center crop
      const imageDataCenterCrop = utils.centerCrop(ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height))
      const ctxCenterCrop = document.getElementById('input-canvas-centercrop').getContext('2d')
      ctxCenterCrop.canvas.width = imageDataCenterCrop.width
      ctxCenterCrop.canvas.height = imageDataCenterCrop.height
      ctxCenterCrop.putImageData(imageDataCenterCrop, 0, 0)

      // scaled to 28 x 28
      const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
      ctxScaled.save()
      ctxScaled.scale(28 / ctxCenterCrop.canvas.width, 28 / ctxCenterCrop.canvas.height)
      ctxScaled.clearRect(0, 0, ctxCenterCrop.canvas.width, ctxCenterCrop.canvas.height)
      ctxScaled.drawImage(document.getElementById('input-canvas-centercrop'), 0, 0)
      const imageDataScaled = ctxScaled.getImageData(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      ctxScaled.restore()

      // process image data for model input
      const { data } = imageDataScaled
      this.input = new Float32Array(784)
      for (let i = 0, len = data.length; i < len; i += 4) {
        this.input[i / 4] = data[i + 3] / 255
      }

      this.model.predict({ input: this.input }).then(outputData => {
        this.output = outputData.output
        this.getIntermediateResults()
      })
    }, 200, { leading: true, trailing: true }),

    getIntermediateResults: function () {
      let results = []
      for (let [name, layer] of this.model.modelLayersMap.entries()) {
        if (name === 'input') continue

        const layerClass = layer.layerClass || ''

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

/* global Vue, loadImage */
import './inception-v3.css'

import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import filter from 'lodash/filter'
import * as utils from './utils'
import { IMAGE_URLS } from './image-urls'
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from './inception-v3-arch'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/inception_v3/inception_v3.json',
  weights: '/demos/data/inception_v3/inception_v3_weights.buf',
  metadata: '/demos/data/inception_v3/inception_v3_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'demos/data/inception_v3/inception_v3.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/inception_v3/inception_v3_weights.buf',
  metadata: 'demos/data/inception_v3/inception_v3_metadata.json'
}
const MODEL_CONFIG = {
  filepaths: (process.env.NODE_ENV === 'production') ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV
}

/**
 *
 * VUE COMPONENT
 *
 */
export const InceptionV3 = Vue.extend({
  props: ['hasWebgl'],

  template: require('raw!./inception-v3.template.html'),

  data: function () {
    return {
      showInfoPanel: true,
      useGpu: this.hasWebgl,
      model: new KerasJS.Model(Object.assign({ gpu: this.hasWebgl, layerCallPauses: true }, MODEL_CONFIG)),
      modelLoading: true,
      modelRunning: false,
      imageURLInput: null,
      imageURLSelect: null,
      imageURLSelectList: IMAGE_URLS,
      imageLoading: false,
      imageLoadingError: false,
      output: null,
      architectureDiagram: ARCHITECTURE_DIAGRAM,
      architectureConnections: ARCHITECTURE_CONNECTIONS,
      architectureDiagramPaths: [],
      showComputationFlow: true
    }
  },

  computed: {
    loadingProgress: function () {
      return this.model.getLoadingProgress()
    },
    architectureDiagramRows: function () {
      let rows = []
      for (let row = 0; row < 112; row++) {
        let cols = []
        for (let col = 0; col < 4; col++) {
          cols.push(filter(this.architectureDiagram, { row, col }))
        }
        rows.push(cols)
      }
      return rows
    },
    layersWithResults: function () {
      // store as computed property for reactivity
      return this.model.layersWithResults
    },
    outputClasses: function () {
      if (!this.output) {
        let empty = []
        for (let i = 0; i < 5; i++) {
          empty.push({ name: '-', probability: 0 })
        }
        return empty
      }
      return utils.imagenetClassesTopK(this.output, 5)
    }
  },

  ready: function () {
    this.model.ready().then(() => {
      this.modelLoading = false

      this.architectureDiagramPaths = []
      setTimeout(() => {
        this.architectureConnections.forEach(conn => {
          const containerElem = document.getElementsByClassName('architecture-container')[0]
          const fromElem = document.getElementById(conn.from)
          const toElem = document.getElementById(conn.to)
          const containerElemCoords = containerElem.getBoundingClientRect()
          const fromElemCoords = fromElem.getBoundingClientRect()
          const toElemCoords = toElem.getBoundingClientRect()
          const xContainer = containerElemCoords.left
          const yContainer = containerElemCoords.top
          const xFrom = fromElemCoords.left + fromElemCoords.width / 2 - xContainer
          const yFrom = fromElemCoords.top + fromElemCoords.height / 2 - yContainer
          const xTo = toElemCoords.left + toElemCoords.width / 2 - xContainer
          const yTo = toElemCoords.top + toElemCoords.height / 2 - yContainer

          let path = `M${xFrom},${yFrom} L${xTo},${yTo}`
          if (conn.corner === 'top-right') {
            path = `M${xFrom},${yFrom} L${xTo - 10},${yFrom} Q${xTo},${yFrom} ${xTo},${yFrom + 10} L${xTo},${yTo}`
          } else if (conn.corner === 'bottom-left') {
            path = `M${xFrom},${yFrom} L${xFrom},${yTo - 10} Q${xFrom},${yTo} ${xFrom + 10},${yTo} L${xTo},${yTo}`
          } else if (conn.corner === 'top-left') {
            path = `M${xFrom},${yFrom} L${xTo + 10},${yFrom} Q${xTo},${yFrom} ${xTo},${yFrom + 10} L${xTo},${yTo}`
          } else if (conn.corner === 'bottom-right') {
            path = `M${xFrom},${yFrom} L${xFrom},${yFrom + 20} Q${xFrom},${yFrom + 30} ${xFrom - 10},${yFrom + 30} L${xTo + 10},${yFrom + 30} Q${xTo},${yFrom + 30} ${xTo},${yFrom + 40} L${xTo},${yTo}`
          }

          this.architectureDiagramPaths.push(path)
        })
      }, 1000)
    })
  },

  methods: {

    closeInfoPanel: function () {
      this.showInfoPanel = false
    },

    toggleGpu: function () {
      this.model.toggleGpu(!this.useGpu)
    },

    toggleComputationFlow: function () {
      this.model.layerCallPauses = !this.showComputationFlow
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
      if (!url) {
        this.clearAll()
        return
      }

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
          maxWidth: 299,
          maxHeight: 299,
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
      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.divseq(dataTensor, 255)
      ops.subseq(dataTensor, 0.5)
      ops.mulseq(dataTensor, 2)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))

      const inputData = {
        'input_1': dataProcessedTensor.data
      }
      this.model.predict(inputData).then(outputData => {
        this.output = outputData['predictions']
        this.modelRunning = false
      })
    },

    clearAll: function () {
      this.modelRunning = false
      this.imageURLInput = null
      this.imageURLSelect = null
      this.imageLoading = false
      this.imageLoadingError = false
      this.output = null

      this.model.layersWithResults = []

      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    }
  }
})

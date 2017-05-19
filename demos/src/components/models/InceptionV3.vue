<template>
  <div class="demo inception-v3">
    <div class="title">
      <span>Inception V3, trained on ImageNet</span>
      <mdl-spinner v-if="modelLoading && loadingProgress < 100"></mdl-spinner>
    </div>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>
    <div class="info-panel" v-if="showInfoPanel">
      <div class="info-panel-text">
        Note that ~100 MB of weights must be loaded. We use the Keras architecture from <a target="_blank" href="https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py">here</a> and pretrained weights from <a target="_blank" href="https://github.com/fchollet/deep-learning-models">here</a>. Enter any valid image URL as input to the network. You can also select from a list of prepopulated image URLs. The endpoint must have CORS enabled, to enable us to extract the numeric data from the canvas element, so not all URLs will work. Imgur and <a target="_blank" href="https://www.flickr.com/search/?text=&license=2%2C3%2C4%2C5%2C6%2C9&sort=interestingness-desc">Flickr creative commons</a> all work, and are good places to start. After running the network, the top-5 classes are displayed. Keep in mind also we are limited to the <a target="_blank" href="https://github.com/transcranial/keras-js/blob/master/demos/src/utils/imagenet.js">1,000 classes of ImageNet</a>. Keep in mind that this is image classification and not object detection, so the network is forced to output a single class through softmax. Best results are on images where the classification target spans a large portion of the image. All computation performed entirely in your browser. Toggling GPU on should offer significant speedups compared to CPU. Running the network may still take several seconds (optimizations to come). With "show computational flow" toggled, computation through the network will be shown in the architecture diagram (scroll down as computation is performed layer by layer). Turning this feature off will also speed up computation.
      </div>
      <div class="info-panel-close">
        <div class="info-panel-close-btn" @click="closeInfoPanel"><i class="material-icons">close</i>CLOSE</div>
      </div>
    </div>
    <div class="top-container" v-if="!modelLoading">
      <div class="input-container">
        <div class="input-label">Enter a valid image URL or select an image from the dropdown:</div>
        <div class="image-url">
          <mdl-textfield
            floating-label="enter image url"
            v-model="imageURLInput"
            spellcheck="false"
            @keyup.native.enter="onImageURLInputEnter"
          ></mdl-textfield>
          <span>or</span>
          <mdl-select
            label="select image"
            id="image-url-select"
            v-model="imageURLSelect"
            :options="imageURLSelectList"
            style="width:200px;"
          ></mdl-select>
        </div>
      </div>
      <div class="controls">
        <mdl-switch v-model="useGpu" :disabled="modelLoading || modelRunning || !hasWebgl">Use GPU</mdl-switch>
        <mdl-switch v-model="showComputationFlow" :disabled="modelLoading || modelRunning">Show computation flow</mdl-switch>
      </div>
    </div>
    <div class="columns input-output" v-if="!modelLoading">
      <div class="column input-column">
        <div class="loading-indicator">
          <mdl-spinner v-if="imageLoading || modelRunning"></mdl-spinner>
          <div class="error" v-if="imageLoadingError">Error loading URL</div>
        </div>
        <div class="canvas-container">
          <canvas id="input-canvas" width="299" height="299"></canvas>
        </div>
      </div>
      <div class="column output-column">
        <div class="output">
          <div class="output-class"
            :class="{ predicted: i === 0 && outputClasses[i].probability.toFixed(2) > 0 }"
            v-for="i in [0, 1, 2, 3, 4]"
          >
            <div class="output-label">{{ outputClasses[i].name }}</div>
            <div class="output-bar"
              :style="{width: `${Math.round(100 * outputClasses[i].probability)}px`, background: `rgba(27, 188, 155, ${outputClasses[i].probability.toFixed(2)})` }"
            ></div>
            <div class="output-value">{{ Math.round(100 * outputClasses[i].probability) }}%</div>
          </div>
        </div>
      </div>
    </div>
    <div class="architecture-container" v-if="!modelLoading">
      <div v-for="(row, rowIndex) in architectureDiagramRows" :key="`row-${rowIndex}`" class="layers-row">
        <div v-for="layers in row" class="layer-column">
          <div v-for="layer in layers" :key="`layer-${layer.name}`"
            v-if="layer.className"
            class="layer"
            :class="{ 'has-result': layersWithResults.includes(layer.name) }"
            :id="layer.name"
          >
            <div class="layer-class-name">{{ layer.className }}</div>
            <div class="layer-details"> {{ layer.details }}</div>
          </div>
        </div>
      </div>
      <svg class="architecture-connections" width="100%" height="100%">
        <g>
          <path v-for="(path, pathIndex) in architectureDiagramPaths" :key="`path-${pathIndex}`" :d="path" />
        </g>
      </svg>
    </div>
  </div>
</template>

<script>
import loadImage from 'blueimp-load-image'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import filter from 'lodash/filter'
import * as utils from '../../utils'
import { IMAGE_URLS } from '../../data/sample-image-urls'
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from '../../data/inception-v3-arch'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/inception_v3/inception_v3.json',
  weights: '/demos/data/inception_v3/inception_v3_weights.buf',
  metadata: '/demos/data/inception_v3/inception_v3_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'https://transcranial.github.io/keras-js-demos-data/inception_v3/inception_v3.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/inception_v3/inception_v3_weights.buf',
  metadata: 'https://transcranial.github.io/keras-js-demos-data/inception_v3/inception_v3_metadata.json'
}
const MODEL_CONFIG = { filepaths: process.env.NODE_ENV === 'production' ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV }

export default {
  props: ['hasWebgl'],

  data: function() {
    return {
      showInfoPanel: true,
      useGpu: this.hasWebgl,
      model: new KerasJS.Model( // eslint-disable-line
        Object.assign({ gpu: this.hasWebgl, pipeline: false, layerCallPauses: true }, MODEL_CONFIG)
      ),
      modelLoading: true,
      modelRunning: false,
      imageURLInput: '',
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

  watch: {
    imageURLSelect: function(value) {
      this.imageURLInput = value
      this.loadImageToCanvas(value)
    },
    useGpu: function(value) {
      this.model.toggleGpu(value)
    },
    showComputationFlow: function(value) {
      this.model.layerCallPauses = value
    }
  },

  computed: {
    loadingProgress: function() {
      return this.model.getLoadingProgress()
    },
    architectureDiagramRows: function() {
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
    layersWithResults: function() {
      // store as computed property for reactivity
      return this.model.layersWithResults
    },
    outputClasses: function() {
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

  mounted: function() {
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
    closeInfoPanel: function() {
      this.showInfoPanel = false
    },
    onImageURLInputEnter: function(e) {
      this.imageURLSelect = null
      this.loadImageToCanvas(e.target.value)
    },
    loadImageToCanvas: function(url) {
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
            this.$nextTick(function() {
              setTimeout(() => {
                this.runModel()
              }, 200)
            })
          }
        },
        { maxWidth: 299, maxHeight: 299, cover: true, crop: true, canvas: true, crossOrigin: 'Anonymous' }
      )
    },
    runModel: function() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

      // data processing
      // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
      // and https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py
      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.divseq(dataTensor, 255)
      ops.subseq(dataTensor, 0.5)
      ops.mulseq(dataTensor, 2)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))

      const inputData = { input_1: dataProcessedTensor.data }
      this.model.predict(inputData).then(outputData => {
        this.output = outputData['predictions']
        this.modelRunning = false
      })
    },
    clearAll: function() {
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
}
</script>

<style scoped>
@import '../../variables.css';

.demo.inception-v3 {
  & .top-container {
    margin: 10px;
    position: relative;
    display: flex;

    & .input-container {
      & .input-label {
        font-family: var(--font-cursive);
        font-size: 16px;
        color: var(--color-lightgray);
        text-align: left;
        user-select: none;
        cursor: default;
      }

      & .image-url {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: flex-start;
        position: relative;

        & span {
          margin: 0 10px;
          font-family: var(--font-cursive);
          font-size: 16px;
          color: var(--color-lightgray);
        }
      }
    }

    & .controls {
      width: 250px;
      margin-left: 40px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;

      & .mdl-switch {
        margin-bottom: 5px;
      }
    }
  }

  & .columns.input-output {
    max-width: 800px;
    margin: 0 auto;

    & .column {
      display: flex;
      align-items: center;
      justify-content: center;
    }

    & .column.input-column {
      position: relative;

      & .loading-indicator {
        position: absolute;
        top: 0;
        left: -10px;
        display: flex;
        flex-direction: column;
        align-self: flex-start;

        & .mdl-spinner {
          margin: 20px;
          align-self: center;
        }

        & .error {
          color: var(--color-error);
          font-size: 14px;
          font-family: var(--font-sans-serif);
          margin: 20px;
        }
      }

      & .canvas-container {
        display: inline-flex;
        justify-content: flex-end;

        & canvas {
          background: white;
        }
      }
    }

    & .column.output-column {
      & .output {
        width: 370px;
        height: 160px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;

        & .output-class {
          display: flex;
          flex-direction: row;
          align-items: center;
          justify-content: center;
          padding: 6px 0;

          & .output-label {
            text-align: right;
            width: 200px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: var(--font-monospace);
            font-size: 18px;
            color: var(--color-lightgray);
            padding: 0 6px;
            border-right: 2px solid var(--color-green-lighter);
          }

          & .output-bar {
            height: 8px;
            transition: width 0.2s ease-out;
          }

          & .output-value {
            text-align: left;
            margin-left: 5px;
            font-family: var(--font-monospace);
            font-size: 14px;
            color: var(--color-lightgray);
          }
        }

        & .output-class.predicted {
          & .output-label {
            color: var(--color-green);
            border-left-color: var(--color-green);
          }

          & .output-value {
            color: var(--color-green);
          }
        }
      }
    }
  }

  & .architecture-container {
    min-width: 800px;
    max-width: 1200px;
    margin: 0 auto;
    position: relative;

    & .layers-row {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;
      margin-bottom: 5px;
      position: relative;
      z-index: 1;

      & .layer-column {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 5px;

        & .layer {
          display: inline-block;
          background: white;
          border: 2px solid white;
          border-radius: 5px;
          padding: 2px 10px 0px;
          margin: 3px;

          & .layer-class-name {
            color: var(--color-green);
            font-size: 14px;
            font-weight: bold;
          }

          & .layer-details {
            color: #999999;
            font-size: 12px;
            font-weight: bold;
          }
        }

        & .layer.has-result {
          border-color: var(--color-green);
        }
      }
    }

    & .architecture-connections {
      position: absolute;
      top: 0;
      left: 0;
      z-index: 0;

      & path {
        stroke-width: 4px;
        stroke: #AAAAAA;
        fill: none;
      }
    }
  }
}
</style>

<template>
  <div class="demo squeezenet-v1">
    <div class="title">
      <span>SqueezeNet v1.1, trained on ImageNet</span>
    </div>
    <mdl-spinner v-if="modelLoading && loadingProgress < 100"></mdl-spinner>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
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
          <canvas id="input-canvas" width="227" height="227"></canvas>
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
        <div v-for="layer in row" :key="`layer-${layer.name}`" class="layer-column">
          <div
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
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from '../../data/squeezenet-v1.1-arch'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/squeezenet_v1.1/squeezenet_v1.1.json',
  weights: '/demos/data/squeezenet_v1.1/squeezenet_v1.1_weights.buf',
  metadata: '/demos/data/squeezenet_v1.1/squeezenet_v1.1_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1_weights.buf',
  metadata: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1_metadata.json'
}
const MODEL_CONFIG = { filepaths: process.env.NODE_ENV === 'production' ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV }

export default {
  props: ['hasWebgl'],

  data: function() {
    return {
      useGpu: this.hasWebgl,
      showComputationFlow: true,
      model: new KerasJS.Model(
        Object.assign({ gpu: this.hasWebgl, pipeline: false, layerCallPauses: true }, MODEL_CONFIG)
      ), // eslint-disable-line
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
      architectureDiagramPaths: []
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
      const rows = []
      for (let row = 0; row < 50; row++) {
        rows.push(filter(this.architectureDiagram, { row }))
      }
      return rows
    },
    layersWithResults: function() {
      // store as computed property for reactivity
      return this.model.layersWithResults
    },
    outputClasses: function() {
      if (!this.output) {
        const empty = []
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
      this.$nextTick(() => {
        this.drawArchitectureDiagramPaths()
      })
    })
  },

  methods: {
    drawArchitectureDiagramPaths: function() {
      this.architectureDiagramPaths = []
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
          path = `M${xFrom},${yFrom} L${xFrom},${yTo - 10} Q${xFrom},${yTo} ${xFrom - 10},${yTo} L${xTo},${yTo}`
        }

        this.architectureDiagramPaths.push(path)
      })
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
        { maxWidth: 227, maxHeight: 227, cover: true, crop: true, canvas: true, crossOrigin: 'Anonymous' }
      )
    },
    runModel: function() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

      // data processing
      // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.subseq(dataTensor.pick(null, null, 2), 103.939)
      ops.subseq(dataTensor.pick(null, null, 1), 116.779)
      ops.subseq(dataTensor.pick(null, null, 0), 123.68)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 2))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 0))

      const inputData = { input_1: dataProcessedTensor.data }
      this.model.predict(inputData).then(outputData => {
        this.output = outputData['loss']
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

.demo.squeezenet-v1 {
  & .top-container {
    margin: 10px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;

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
          background: whitesmoke;
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
    min-width: 700px;
    max-width: 900px;
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
          background: whitesmoke;
          border: 2px solid whitesmoke;
          border-radius: 5px;
          padding: 2px 10px 0px;

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

<template>
  <div class="demo">
    <v-progress-circular v-if="modelLoading && loadingProgress < 100" indeterminate color="primary" />
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">Loading...{{ loadingProgress }}%</div>
    <div v-if="!modelLoading" class="ui-container">
      <v-layout row justify-center class="input-label">
        Enter a valid image URL or select an image from the dropdown:
      </v-layout>
      <v-layout row wrap justify-center align-center>
        <v-flex xs7 md5>
          <v-text-field
            v-model="imageURLInput"
            label="enter image url"
            @keyup.native.enter="onImageURLInputEnter"
          ></v-text-field>
        </v-flex>
        <v-flex xs1 class="input-label text-xs-center">or</v-flex>
        <v-flex xs5 md3>
          <v-select
            v-model="imageURLSelect"
            :items="imageURLSelectList"
            label="select image"
            max-height="500"
          ></v-select>
        </v-flex>
        <v-flex xs2 md2 class="controls">
          <v-switch label="use GPU" 
            v-model="useGPU" :disabled="modelLoading || modelRunning || !hasWebGL" color="primary"
          ></v-switch>
        </v-flex>
      </v-layout>
      <v-layout row wrap justify-center class="image-panel elevation-1">
        <div v-if="imageLoading || modelRunning" class="loading-indicator">
          <v-progress-circular indeterminate color="primary" />
        </div>
        <div v-if="imageLoadingError" class="error-message">Error loading URL</div>
        <v-flex sm5 md3 align-flex-end class="visualization">
          <v-select
            v-model="visualizationSelect"
            :items="visualizationSelectList"
            label="visualization"
            max-height="500"
          ></v-select>
          <v-select
            v-show="visualizationSelect !== 'None'"
            v-model="colormapSelect"
            :items="colormapSelectList"
            label="colormap"
            max-height="500"
          ></v-select>
          <div 
            v-show="visualizationSelect !== 'None' && colormapSelect !== 'transparency'"
            class="colormap-alpha"
          >
            <label>{{ `opacity: ${colormapAlpha}` }}</label>
            <v-slider v-model="colormapAlpha" min="0" max="1" step="0.01"></v-slider>
          </div>
        </v-flex>
        <v-flex sm5 md3 class="canvas-container">
          <canvas id="input-canvas" width="227" height="227"
            v-on:mouseenter="showVis = true"
            v-on:mouseleave="showVis = false"
          ></canvas>
          <transition name="fade">
            <div v-show="showVis">
              <canvas id="visualization-canvas" width="227" height="227"
                :style="{ opacity: colormapSelect === 'transparency' ? 1 : colormapAlpha }"
              ></canvas>
            </div>
          </transition>
        </v-flex>
        <v-flex sm6 md4 class="output-container">
          <div class="inference-time">
            <span>inference time: </span>
            <span v-if="inferenceTime > 0" class="inference-time-value">{{ inferenceTime.toFixed(1) }} ms </span>
            <span v-if="inferenceTime > 0" class="inference-time-value">({{ (1000 / inferenceTime).toFixed(1) }} fps)</span>
            <span v-else>-</span>
          </div>
          <div v-for="i in [0, 1, 2, 3, 4]" :key="i"
            class="output-class" :class="{ predicted: i === 0 && outputClasses[i].probability.toFixed(2) > 0 }"
          >
            <div class="output-label">{{ outputClasses[i].name }}</div>
            <div class="output-bar"
              :style="{width: `${Math.round(100 * outputClasses[i].probability)}px`, background: `rgba(27, 188, 155, ${outputClasses[i].probability.toFixed(2)})` }"
            ></div>
            <div class="output-value">{{ Math.round(100 * outputClasses[i].probability) }}%</div>
          </div>
        </v-flex>
      </v-layout>
    </div>
    <div v-if="!modelLoading" v-resize="drawArchitectureDiagramPaths" class="architecture-container">
      <div v-for="(row, rowIndex) in architectureDiagramRows" :key="`row-${rowIndex}`" class="layers-row">
        <div v-for="layer in row" :key="`layer-${layer.name}`" class="layer-column">
          <div
            v-if="layer.className"
            class="layer"
            :class="{ 'has-output': finishedLayerNames.includes(layer.name) }"
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
import _ from 'lodash'
import * as utils from '../../utils'
import { IMAGE_URLS } from '../../data/sample-image-urls'
import { COLORMAPS } from '../../data/colormaps'
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from '../../data/squeezenet-v1.1-arch'

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1.bin'
const MODEL_FILEPATH_DEV = '/demos/data/squeezenet_v1.1/squeezenet_v1.1.bin'

export default {
  props: ['hasWebGL'],

  data() {
    return {
      useGPU: this.hasWebGL,
      model: new KerasJS.Model({
        filepath: process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV,
        gpu: this.hasWebGL,
        visualizations: ['CAM']
      }),
      modelLoading: true,
      modelRunning: false,
      inferenceTime: null,
      imageURLInput: '',
      imageURLSelect: null,
      imageURLSelectList: IMAGE_URLS,
      imageLoading: false,
      imageLoadingError: false,
      visualizationSelect: 'CAM',
      visualizationSelectList: [{ text: 'None', value: 'None' }, { text: 'Class Activation Mapping', value: 'CAM' }],
      colormapSelect: 'transparency',
      colormapSelectList: COLORMAPS,
      colormapAlpha: 0.7,
      showVis: false,
      output: null,
      architectureDiagram: ARCHITECTURE_DIAGRAM,
      architectureConnections: ARCHITECTURE_CONNECTIONS,
      architectureDiagramPaths: []
    }
  },

  watch: {
    imageURLSelect(newVal) {
      this.imageURLInput = newVal
      this.loadImageToCanvas(newVal)
    },
    useGPU(newVal) {
      this.model.toggleGPU(newVal)
    },
    visualizationSelect(newVal) {
      if (newVal === 'None') {
        this.showVis = false
      } else {
        this.updateVis()
      }
    },
    colormapSelect() {
      this.updateVis()
    }
  },

  computed: {
    loadingProgress() {
      return this.model.getLoadingProgress()
    },
    architectureDiagramRows() {
      const rows = []
      for (let row = 0; row < 50; row++) {
        rows.push(_.filter(this.architectureDiagram, { row }))
      }
      return rows
    },
    finishedLayerNames() {
      // store as computed property for reactivity
      return this.model.finishedLayerNames
    },
    outputClasses() {
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

  mounted() {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(() => {
        this.drawArchitectureDiagramPaths()
      })
    })
  },

  methods: {
    drawArchitectureDiagramPaths: _.debounce(function() {
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
    }, 100),
    onImageURLInputEnter(e) {
      this.imageURLSelect = null
      this.loadImageToCanvas(e.target.value)
    },
    loadImageToCanvas(url) {
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
    runModel() {
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
        this.inferenceTime = this.model.predictStats.forwardPass
        this.output = outputData['loss']
        this.modelRunning = false
        this.updateVis()
      })
    },
    updateVis() {
      if (!this.output || !this.model.visMap.has(this.visualizationSelect)) return

      const vis = this.model.visMap.get(this.visualizationSelect)
      const height = vis.height
      const width = vis.width
      const imageDataArr = ndarray(new Uint8ClampedArray(height * width * 4), [height, width, 4])

      if (this.colormapSelect === 'transparency') {
        const alpha = ndarray(new Float32Array(vis.data), [height, width])
        ops.mulseq(alpha, -255)
        ops.addseq(alpha, 255)
        ops.assign(imageDataArr.pick(null, null, 3), alpha)
      } else {
        const colormap = this.colormapSelect
        for (let i = 0, len = vis.data.length; i < len; i++) {
          const rgb = colormap(vis.data[i]).rgb()
          imageDataArr.data[4 * i] = rgb[0]
          imageDataArr.data[4 * i + 1] = rgb[1]
          imageDataArr.data[4 * i + 2] = rgb[2]
          imageDataArr.data[4 * i + 3] = 255
        }
      }

      const imageData = new ImageData(width, height)
      imageData.data.set(imageDataArr.data)
      const ctx = document.getElementById('visualization-canvas').getContext('2d')
      ctx.putImageData(imageData, 0, 0)
    },
    clearAll() {
      this.modelRunning = false
      this.inferenceTime = null
      this.imageURLInput = null
      this.imageURLSelect = null
      this.imageLoading = false
      this.imageLoadingError = false
      this.output = null

      this.model.finishedLayerNames = []

      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    }
  }
}
</script>

<style scoped lang="postcss">
@import '../../variables.css';

.ui-container {
  font-family: var(--font-monospace);
  margin-bottom: 30px;
}

.input-label {
  font-family: var(--font-cursive);
  font-size: 16px;
  color: var(--color-lightgray);
  text-align: left;
  user-select: none;
  cursor: default;
}

.controls {
  width: 100px;
  margin-left: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.image-panel {
  padding: 30px 20px;
  background-color: whitesmoke;
  position: relative;

  & .loading-indicator {
    position: absolute;
    top: 5px;
    left: 5px;
  }

  & .error-message {
    color: var(--color-error);
    font-size: 12px;
    position: absolute;
    top: 5px;
    left: 5px;
  }
}

.visualization {
  margin-right: 20px;

  & .colormap-alpha {
    position: relative;

    & label {
      position: absolute;
      color: var(--color-darkgray);
      font-size: 10px;
    }
  }
}

.canvas-container {
  position: relative;
  margin: 0 20px;

  & #input-canvas {
    background: #eeeeee;
  }

  & #visualization-canvas {
    pointer-events: none;
    position: absolute;
    top: 0;
    left: 0;
  }
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease-out;
}
.fade-enter,
.fade-leave-to {
  opacity: 0;
}

.output-container {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;

  & .inference-time {
    align-self: center;
    font-family: var(--font-monospace);
    font-size: 14px;
    color: var(--color-lightgray);
    margin-bottom: 10px;

    & .inference-time-value {
      color: var(--color-green);
    }
  }

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
      font-size: 16px;
      color: var(--color-darkgray);
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

.architecture-container {
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
        padding: 2px 5px 0px;

        & .layer-class-name {
          color: var(--color-green);
          font-size: 12px;
          font-weight: bold;
        }

        & .layer-details {
          color: #999999;
          font-size: 11px;
          font-weight: bold;
        }
      }

      & .layer.has-output {
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
      stroke: #aaaaaa;
      fill: none;
    }
  }
}
</style>

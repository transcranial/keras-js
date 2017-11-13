<template>
  <div class="demo mnist-acgan">
    <div class="title">
      <span>Auxiliary Classifier Generative Adversarial Networks (AC-GAN) on MNIST</span>
    </div>
    <mdl-spinner v-if="modelLoading && loadingProgress < 100"></mdl-spinner>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>
    <div class="columns input-output" v-if="!modelLoading">
      <div class="column is-half input-column">
        <div class="input-container">
          <canvas id="noise-canvas" width="120" height="120"></canvas>
          <div class="input-items">
            <div v-for="n in 10"
              class="digit-select" :class="{ active: digit === n - 1 }"
              @click="selectDigit(n - 1)"
            >{{ n - 1 }}</div>
            <div
              class="noise-btn"
              @click="onGenerateNewNoise"
            >Generate New Noise</div>
          </div>
        </div>
      </div>
      <div class="column output-column">
        <div class="output">
          <canvas id="output-canvas-scaled" width="140" height="140"></canvas>
          <canvas id="output-canvas" width="28" height="28" style="display:none;"></canvas>
        </div>
      </div>
      <div class="column controls-column">
        <mdl-switch v-model="useGPU" :disabled="modelLoading || !hasWebGL">use GPU</mdl-switch>
      </div>
    </div>
    <div class="architecture-container" v-if="!modelLoading">
      <div v-for="(row, rowIndex) in architectureDiagramRows" :key="`row-${rowIndex}`" class="layers-row">
        <div v-for="layer in row" :key="`layer-${layer.name}`" class="layer-column">
          <div v-if="layer.className" class="layer" :id="layer.name">
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
import * as utils from '../../utils'
import filter from 'lodash/filter'
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from '../../data/mnist-acgan-arch'

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/mnist_acgan/mnist_acgan.bin'
const MODEL_FILEPATH_DEV = '/demos/data/mnist_acgan/mnist_acgan.bin'

export default {
  props: ['hasWebGL'],

  data() {
    return {
      useGPU: this.hasWebGL,
      digit: 6,
      noiseVector: [],
      model: new KerasJS.Model({
        filepath: process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV,
        gpu: this.hasWebGL
      }),
      modelLoading: true,
      output: new Float32Array(28 * 28),
      architectureDiagram: ARCHITECTURE_DIAGRAM,
      architectureConnections: ARCHITECTURE_CONNECTIONS,
      architectureDiagramPaths: []
    }
  },

  watch: {
    useGPU(value) {
      this.model.toggleGPU(value)
    }
  },

  computed: {
    loadingProgress() {
      return this.model.getLoadingProgress()
    },
    architectureDiagramRows() {
      const rows = []
      for (let row = 0; row < 12; row++) {
        rows.push(filter(this.architectureDiagram, { row }))
      }
      return rows
    }
  },

  created() {
    this.createNoise()
  },

  mounted() {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(() => {
        this.drawArchitectureDiagramPaths()
        this.runModel()
        this.drawNoise()
      })
    })
  },

  methods: {
    drawArchitectureDiagramPaths() {
      this.architectureDiagramPaths = []
      this.$nextTick(() => {
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
      })
    },
    selectDigit(digit) {
      this.digit = digit
      this.runModel()
    },
    createNoise() {
      const latentSize = 100
      const noiseVector = []
      for (let i = 0; i < latentSize; i++) {
        // uniform random between -1 and 1
        noiseVector.push(2 * Math.random() - 1)
      }
      this.noiseVector = noiseVector
    },
    runModel() {
      const inputData = {
        input_2: new Float32Array(this.noiseVector),
        input_3: new Float32Array([this.digit])
      }
      this.model.predict(inputData).then(outputData => {
        this.output = outputData['conv2d_7']
        this.drawOutput()
      })
    },
    drawNoise() {
      // draw noise visualization on canvas
      const ctx = document.getElementById('noise-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      for (let x = 0; x < 10; x++) {
        for (let y = 0; y < 10; y++) {
          ctx.fillStyle = `rgba(0, 0, 0, ${(this.noiseVector[10 * x + y] + 1) / 2})`
          // scale 12x
          ctx.fillRect(12 * x, 12 * y, 12, 12)
        }
      }
    },
    drawOutput() {
      const ctx = document.getElementById('output-canvas').getContext('2d')
      const image = utils.image2Darray(this.output, 28, 28, [0, 0, 0])
      ctx.putImageData(image, 0, 0)

      // scale up
      const ctxScaled = document.getElementById('output-canvas-scaled').getContext('2d')
      ctxScaled.save()
      ctxScaled.scale(140 / 28, 140 / 28)
      ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      ctxScaled.drawImage(document.getElementById('output-canvas'), 0, 0)
      ctxScaled.restore()
    },
    onGenerateNewNoise() {
      this.createNoise()
      this.runModel()
      this.drawNoise()
    }
  }
}
</script>

<style scoped>
@import '../../variables.css';

.demo.mnist-acgan {
  & .column {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  & .column.input-column {
    justify-content: flex-end;

    & .input-container {
      margin: 20px;
      position: relative;
      user-select: none;
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;

      & canvas {
        margin: 10px;
      }

      & .input-items {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        width: 280px;

        & .digit-select, & .noise-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 50px;
          height: 50px;
          margin: 2px;
          border: 1px solid var(--color-green-lighter);
          color: var(--color-green);
          font-size: 20px;
          font-weight: bold;
          font-family: var(--font-monospace);
          transition: background-color 0.1s ease;
          cursor: default;

          &.active {
            background-color: var(--color-green);
            color: white;
          }

          &:hover:not(.active) {
            background-color: var(--color-green-lighter);
            cursor: pointer;
          }
        }

        & .noise-btn {
          width: 266px;
          font-size: 14px;
        }
      }

      & .canvas-container {
        position: relative;
        display: inline-flex;
        justify-content: flex-end;
        margin: 10px 0;
        border: 15px solid var(--color-green-lighter);
        transition: border-color 0.2s ease-in;

        & canvas {
          background: whitesmoke;
        }
      }
    }
  }

  & .column.output-column {
    & .output {
      border-radius: 10px;
      border: 1px solid gray;
      overflow: hidden;

      & canvas {
        background: whitesmoke;
      }
    }
  }

  & .column.controls-column {
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;

    & .mdl-switch {
      width: auto;
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
          border: 2px solid var(--color-green);
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

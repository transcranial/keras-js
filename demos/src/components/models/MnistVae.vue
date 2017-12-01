<template>
  <div class="demo">
    <transition name="fade">
      <model-status v-if="modelLoading || modelInitializing" 
        :modelLoading="modelLoading"
        :modelLoadingProgress="modelLoadingProgress"
        :modelInitializing="modelInitializing"
        :modelInitProgress="modelInitProgress"
      ></model-status>
    </transition>
    <v-alert outline color="error" icon="priority_high" :value="!hasWebGL">
      Note: this browser does not support WebGL 2 or the features necessary to run in GPU mode.
    </v-alert>
    <v-layout row wrap justify-center>
      <v-flex sm2 md1>
        <div class="controls-column">
          <div class="control">
            <v-switch 
              :disabled="modelLoading || modelInitializing || !hasWebGL"
              label="use GPU"
              v-model="useGPU"
              color="primary"
            ></v-switch>
          </div>
          <div class="coordinates">
            <div class="coordinates-x">x: {{ inputCoordinates[0] < 0 ? inputCoordinates[0].toFixed(2) : inputCoordinates[0].toFixed(3) }}</div>
            <div class="coordinates-y">y: {{ inputCoordinates[1] < 0 ? inputCoordinates[1].toFixed(2) : inputCoordinates[1].toFixed(3) }}</div>
          </div>
        </div>
      </v-flex>
      <v-flex sm6 md4>
        <div class="input-column">
          <div class="input-container">
            <div class="input-label">Move around the latent space <span class="arrow">⤸</span></div>
            <div class="canvas-container">
              <canvas
                id="input-canvas" width="200" height="200"
                @mouseenter="activateCrosshairs"
                @mouseleave="deactivateCrosshairs"
                @mousemove="selectCoordinates"
                @click="selectCoordinates"
                @touchend="selectCoordinates"
              ></canvas>
              <div class="axis x-axis">
                <span>-1</span>
                <span>x</span>
                <span>1</span>
              </div>
              <div class="axis y-axis">
                <span>-1</span>
                <span>y</span>
                <span>1</span>
              </div>
            </div>
          </div>
        </div>
      </v-flex>
      <v-flex sm4 md2>
        <div class="output-column">
          <div class="output">
            <canvas id="output-canvas-scaled" width="140" height="140"></canvas>
            <canvas id="output-canvas" width="28" height="28" style="display:none;"></canvas>
          </div>
        </div>
      </v-flex>
    </v-layout>
    <div class="layer-outputs-container" v-if="!modelLoading">
      <div class="bg-line"></div>
      <div
        v-for="(layerOutput, layerIndex)  in layerOutputImages"
        :key="`intermediate-output-${layerIndex}`"
        class="layer-output"
      >
        <div class="layer-output-heading">
          <span class="layer-class">{{ layerOutput.layerClass }}</span>
          <span> {{ layerDisplayConfig[layerOutput.name].heading }}</span>
        </div>
        <div class="layer-output-canvas-container">
          <canvas v-for="(image, index) in layerOutput.images"
            :key="`intermediate-output-${layerIndex}-${index}`"
            :id="`intermediate-output-${layerIndex}-${index}`"
            :width="image.width"
            :height="image.height"
            style="display:none;"
          ></canvas>
          <canvas v-for="(image, index) in layerOutput.images"
            :key="`intermediate-output-${layerIndex}-${index}-scaled`"
            :id="`intermediate-output-${layerIndex}-${index}-scaled`"
            :width="layerDisplayConfig[layerOutput.name].scalingFactor * image.width"
            :height="layerDisplayConfig[layerOutput.name].scalingFactor * image.height"
          ></canvas>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import _ from 'lodash'
import { tensorUtils } from '../../utils'
import ModelStatus from '../common/ModelStatus'

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/mnist_vae/mnist_vae.bin'
const MODEL_FILEPATH_DEV = '/demos/data/mnist_vae/mnist_vae.bin'

const LAYER_DISPLAY_CONFIG = {
  dense_6: { heading: 'input dimensions = 2, output dimensions = 128, ReLU activation', scalingFactor: 2 },
  dense_7: { heading: 'ReLU activation, output dimensions = 25088 (64 x 14 x 14)', scalingFactor: 2 },
  reshape_2: { heading: '', scalingFactor: 2 },
  conv2d_transpose_4: { heading: '64 3x3 filters, padding same, 1x1 strides, ReLU activation', scalingFactor: 2 },
  conv2d_transpose_5: { heading: '64 3x3 filters, padding same, 1x1 strides, ReLU activation', scalingFactor: 2 },
  conv2d_transpose_6: { heading: '64 2x2 filters, padding valid, 2x2 strides, ReLU activation', scalingFactor: 2 },
  conv2d_6: { heading: '1 2x2 filters, padding same, 1x1 strides, sigmoid activation', scalingFactor: 2 }
}

export default {
  props: ['hasWebGL'],

  components: { ModelStatus },

  created() {
    // store module on component instance as non-reactive object
    this.model = new KerasJS.Model({
      filepath: process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV,
      gpu: this.hasWebGL,
      transferLayerOutputs: true
    })

    this.model.events.on('loadingProgress', this.handleLoadingProgress)
    this.model.events.on('initProgress', this.handleInitProgress)
  },

  async mounted() {
    await this.model.ready()
    await this.$nextTick()
    this.drawPosition()
    this.getIntermediateOutputs()
    this.runModel()
  },

  beforeDestroy() {
    this.model.cleanup()
    this.model.events.removeAllListeners()
  },

  data() {
    return {
      useGPU: this.hasWebGL,
      modelLoading: true,
      modelLoadingProgress: 0,
      modelInitializing: true,
      modelInitProgress: 0,
      output: new Float32Array(28 * 28),
      crosshairsActivated: false,
      inputCoordinates: [-0.3, -0.6],
      position: [35, 20],
      layerOutputImages: [],
      layerDisplayConfig: LAYER_DISPLAY_CONFIG
    }
  },

  watch: {
    useGPU(value) {
      this.model.toggleGPU(value)
    }
  },

  methods: {
    handleLoadingProgress(progress) {
      this.modelLoadingProgress = Math.round(progress)
      if (progress === 100) {
        this.modelLoading = false
      }
    },
    handleInitProgress(progress) {
      this.modelInitProgress = Math.round(progress)
      if (progress === 100) {
        this.modelInitializing = false
      }
    },
    activateCrosshairs() {
      if (this.modelLoading || this.modelInitializing) return
      this.crosshairsActivated = true
    },
    deactivateCrosshairs(e) {
      this.crosshairsActivated = false
      this.draw(e)
    },
    draw(e) {
      const [x, y] = this.getEventCanvasCoordinates(e)
      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, 200, 200)

      this.drawPosition()

      if (this.crosshairsActivated) {
        ctx.strokeStyle = '#1BBC9B'
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, 200)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(200, y)
        ctx.stroke()
      }
    },
    drawPosition() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, 200, 200)
      ctx.fillStyle = 'rgb(0, 0, 0)'
      ctx.beginPath()
      ctx.arc(...this.position, 5, 0, Math.PI * 2, true)
      ctx.closePath()
      ctx.fill()
    },
    getEventCanvasCoordinates(e) {
      let { clientX, clientY } = e
      // for touch event
      if (e.touches && e.touches.length) {
        clientX = e.touches[0].clientX
        clientY = e.touches[0].clientY
      }

      const canvas = document.getElementById('input-canvas')
      const { left, top } = canvas.getBoundingClientRect()
      const [x, y] = [clientX - left, clientY - top]
      return [x, y]
    },
    selectCoordinates: _.throttle(
      function(e) {
        if (this.modelLoading || this.modelInitializing) return
        this.draw(e)
        const [x, y] = this.getEventCanvasCoordinates(e)
        if (!this.model.isRunning) {
          this.position = [x, y]
          this.inputCoordinates = [x * 2 / 200 - 1, y * 2 / 200 - 1]
          this.runModel()
        }
      },
      16,
      { leading: true, trailing: true }
    ),
    async runModel() {
      const inputData = { input_2: new Float32Array(this.inputCoordinates) }
      const outputData = await this.model.predict(inputData)
      this.output = outputData['conv2d_6']
      this.drawOutput()
      this.getIntermediateOutputs()
    },
    drawOutput() {
      const ctx = document.getElementById('output-canvas').getContext('2d')
      const image = tensorUtils.image2Darray(this.output, 28, 28, [0, 0, 0])
      ctx.putImageData(image, 0, 0)

      // scale up
      const ctxScaled = document.getElementById('output-canvas-scaled').getContext('2d')
      ctxScaled.save()
      ctxScaled.scale(140 / 28, 140 / 28)
      ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      ctxScaled.drawImage(document.getElementById('output-canvas'), 0, 0)
      ctxScaled.restore()
    },
    getIntermediateOutputs() {
      const outputs = []
      this.model.modelLayersMap.forEach((layer, name) => {
        if (layer.layerClass === 'InputLayer') return
        let images = []
        if (layer.hasOutput && layer.output && layer.output.tensor.shape.length === 3) {
          images = tensorUtils.unroll3Dtensor(layer.output.tensor)
        } else if (layer.hasOutput && layer.output && layer.output.tensor.shape.length === 2) {
          images = [tensorUtils.image2Dtensor(layer.output.tensor)]
        } else if (layer.hasOutput && layer.output && layer.output.tensor.shape.length === 1) {
          images = [tensorUtils.image1Dtensor(layer.output.tensor)]
        }
        outputs.push({ layerClass: layer.layerClass || '', name, images })
      })
      this.layerOutputImages = outputs
      setTimeout(() => {
        this.showIntermediateOutputs()
      }, 0)
    },
    showIntermediateOutputs() {
      this.layerOutputImages.forEach((output, layerNum) => {
        const scalingFactor = this.layerDisplayConfig[output.name].scalingFactor
        output.images.forEach((image, imageNum) => {
          const ctx = document.getElementById(`intermediate-output-${layerNum}-${imageNum}`).getContext('2d')
          ctx.putImageData(image, 0, 0)
          const ctxScaled = document
            .getElementById(`intermediate-output-${layerNum}-${imageNum}-scaled`)
            .getContext('2d')
          ctxScaled.save()
          ctxScaled.scale(scalingFactor, scalingFactor)
          ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
          ctxScaled.drawImage(document.getElementById(`intermediate-output-${layerNum}-${imageNum}`), 0, 0)
          ctxScaled.restore()
        })
      })
    },
    clearIntermediateOutputs() {
      this.layerOutputImages.forEach((output, layerNum) => {
        const scalingFactor = this.layerDisplayConfig[output.name].scalingFactor
        output.images.forEach((image, imageNum) => {
          const ctxScaled = document
            .getElementById(`intermediate-output-${layerNum}-${imageNum}-scaled`)
            .getContext('2d')
          ctxScaled.save()
          ctxScaled.scale(scalingFactor, scalingFactor)
          ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
          ctxScaled.restore()
        })
      })
    }
  }
}
</script>

<style scoped lang="postcss">
@import '../../variables.css';

.controls-column {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  font-family: var(--font-monospace);
  padding-top: 80px;

  & .control {
    width: 100px;
    margin: 10px auto;
  }

  & .coordinates {
    margin: 10px auto;
    font-size: 18px;
    color: var(--color-lightgray);
  }
}

.input-column {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;

  & .input-container {
    text-align: left;
    margin: 20px;
    position: relative;
    user-select: none;

    & .input-label {
      font-family: var(--font-cursive);
      font-size: 18px;
      color: var(--color-lightgray);

      & span.arrow {
        font-size: 36px;
        color: #cccccc;
        position: absolute;
        left: 260px;
        top: 8px;
      }
    }

    & .canvas-container {
      position: relative;
      display: inline-flex;
      justify-content: flex-end;
      margin: 10px 0 10px 30px;
      border: 15px solid var(--color-green-lighter);
      transition: border-color 0.2s ease-in;

      &:hover {
        border-color: var(--color-green-light);
      }

      & canvas {
        background: whitesmoke;

        &:hover {
          cursor: crosshair;
        }
      }

      & .axis {
        position: absolute;
        cursor: default;
        user-select: none;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: var(--font-monospace);
        font-size: 14px;
        color: var(--color-green);
      }

      & .axis.x-axis {
        right: 0;
        bottom: -45px;
        width: 200px;
        flex-direction: row;
      }

      & .axis.y-axis {
        top: 0;
        left: -55px;
        height: 200px;
        flex-direction: column;
      }
    }
  }
}

.output-column {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 40px;

  & .output {
    margin: 20px 0;
    border-radius: 10px;
    border: 1px solid gray;
    overflow: hidden;

    & canvas {
      background: whitesmoke;
    }
  }
}

.layer-outputs-container {
  position: relative;

  & .bg-line {
    position: absolute;
    z-index: 0;
    top: 0;
    left: 50%;
    background: whitesmoke;
    width: 15px;
    height: 100%;
  }

  & .layer-output {
    position: relative;
    z-index: 1;
    margin: 30px 20px;
    background: whitesmoke;
    border-radius: 10px;
    padding: 20px;
    overflow-x: auto;

    & .layer-output-heading {
      font-size: 1rem;
      color: #999999;
      margin-bottom: 10px;
      display: flex;
      flex-direction: column;
      font-size: 12px;

      & span.layer-class {
        color: var(--color-green);
        font-size: 14px;
        font-weight: bold;
      }
    }

    & .layer-output-canvas-container {
      display: inline-flex;
      flex-wrap: wrap;
      background: whitesmoke;

      & canvas {
        border: 1px solid lightgray;
        margin: 1px;
      }
    }
  }
}

/* vue transition `fade` */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s;
}
.fade-enter,
.fade-leave-to {
  opacity: 0;
}
</style>

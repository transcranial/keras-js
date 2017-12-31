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
        </div>
      </v-flex>
      <v-flex sm8 md6>
        <div class="input-column">
          <div class="input-container">
            <canvas id="noise-canvas" width="120" height="120"></canvas>
            <div class="input-items">
              <div v-for="n in 10" :key="n" class="digit-select" :class="{ active: digit === n - 1 }"
                @click="selectDigit(n - 1)"
              >{{ n - 1 }}</div>
              <div
                class="noise-btn"
                @click="onGenerateNewNoise"
              >Generate New Noise</div>
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
    <architecture-diagram :modelLayersInfo="modelLayersInfo"></architecture-diagram>
  </div>
</template>

<script>
import { tensorUtils } from '../../utils'
import ModelStatus from '../common/ModelStatus'
import ArchitectureDiagram from '../common/ArchitectureDiagram'

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/mnist_acgan/mnist_acgan.bin'
const MODEL_FILEPATH_DEV = '/demos/data/mnist_acgan/mnist_acgan.bin'

export default {
  props: ['hasWebGL'],

  components: { ModelStatus, ArchitectureDiagram },

  created() {
    this.createNoise()
    // store module on component instance as non-reactive object
    this.model = new KerasJS.Model({
      filepath: process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV,
      gpu: this.hasWebGL
    })

    this.model.events.on('loadingProgress', this.handleLoadingProgress)
    this.model.events.on('initProgress', this.handleInitProgress)
  },

  async mounted() {
    await this.model.ready()
    await this.$nextTick()
    this.modelLayersInfo = this.model.modelLayersInfo
    this.runModel()
    this.drawNoise()
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
      modelLayersInfo: [],
      digit: 3,
      noiseVector: [],
      output: new Float32Array(28 * 28)
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
    selectDigit(digit) {
      if (this.modelLoading || this.modelInitializing) return
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
    async runModel() {
      const inputData = {
        input_2: new Float32Array(this.noiseVector),
        input_3: new Float32Array([this.digit])
      }
      const outputData = await this.model.predict(inputData)
      this.output = outputData['conv2d_7']
      this.drawOutput()
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
    onGenerateNewNoise() {
      if (this.modelLoading || this.modelInitializing) return
      this.createNoise()
      this.runModel()
      this.drawNoise()
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
  padding-top: 20px;

  & .control {
    width: 100px;
    margin: 10px 0;
  }
}

.input-column {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;

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

      & .digit-select,
      & .noise-btn {
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

.output-column {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;

  & .output {
    display: inline-flex;
    margin: 20px 0;
    border-radius: 10px;
    border: 1px solid gray;
    overflow: hidden;

    & canvas {
      background: whitesmoke;
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

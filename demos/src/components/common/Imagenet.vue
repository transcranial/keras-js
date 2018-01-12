<template>
  <div class="imagenet">
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
    <div class="ui-container">
      <v-layout row justify-center class="input-label">
        Enter a valid image URL or select an image from the dropdown:
      </v-layout>
      <v-layout row wrap justify-center align-center>
        <v-flex xs7 md5>
          <v-text-field
            v-model="imageURLInput"
            :disabled="modelLoading || modelInitializing"
            label="enter image url"
            :spellcheck="false"
            @keyup.native.enter="onImageURLInputEnter"
          ></v-text-field>
        </v-flex>
        <v-flex xs1 class="input-label text-xs-center">or</v-flex>
        <v-flex xs5 md3>
          <v-select
            v-model="imageURLSelect"
            :disabled="modelLoading || modelInitializing"
            :items="imageURLSelectList"
            label="select image"
            max-height="750"
          ></v-select>
        </v-flex>
        <v-flex xs2 md2 class="controls">
          <v-switch label="use GPU" 
            v-model="useGPU"
            :disabled="modelLoading || modelInitializing || modelRunning || !hasWebGL" 
            color="primary"
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
            :disabled="modelLoading || modelInitializing"
            :items="visualizationSelectList"
            label="visualization"
            max-height="500"
          ></v-select>
          <v-select
            v-show="visualizationSelect !== 'None'"
            v-model="colormapSelect"
            :disabled="modelLoading || modelInitializing"
            :items="colormapSelectList"
            label="colormap"
            max-height="500"
          ></v-select>
          <div 
            v-show="visualizationSelect !== 'None' && colormapSelect !== 'transparency'"
            class="colormap-alpha"
          >
            <label>{{ `opacity: ${colormapAlpha}` }}</label>
            <v-slider 
              v-model="colormapAlpha"
              :disabled="modelLoading || modelInitializing"
              min="0"
              max="1"
              step="0.01"
            ></v-slider>
          </div>
          <div 
            v-show="visualizationSelect !== 'None' && output !== null" 
            class="visualization-instruction"
          >(hover over image to view)</div>
        </v-flex>
        <v-flex sm5 md3 class="canvas-container">
          <canvas id="input-canvas"
            :width="imageSize"
            :height="imageSize"
            v-on:mouseenter="showVis = true"
            v-on:mouseleave="showVis = false"
          ></canvas>
          <transition name="fade">
            <div v-show="showVis">
              <canvas id="visualization-canvas"
                :width="imageSize"
                :height="imageSize"
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
    <architecture-diagram :modelLayersInfo="modelLayersInfo"></architecture-diagram>
  </div>
</template>

<script>
import loadImage from 'blueimp-load-image'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import resample from 'ndarray-resample'
import { imagenetUtils } from '../../utils'
import { IMAGE_URLS } from '../../data/sample-image-urls'
import { COLORMAPS } from '../../data/colormaps'
import ModelStatus from './ModelStatus'
import ArchitectureDiagram from './ArchitectureDiagram'

export default {
  props: {
    modelName: { type: String, required: true },
    modelFilepath: { type: String, required: true },
    hasWebGL: { type: Boolean, required: true },
    imageSize: { type: Number, required: true },
    visualizations: { type: Array, required: true },
    preprocess: { type: Function, required: true }
  },

  components: { ModelStatus, ArchitectureDiagram },

  created() {
    // store module on component instance as non-reactive object
    this.model = new KerasJS.Model({
      filepath: this.modelFilepath,
      gpu: this.hasWebGL,
      visualizations: this.visualizations
    })

    this.model.events.on('loadingProgress', this.handleLoadingProgress)
    this.model.events.on('initProgress', this.handleInitProgress)
  },

  async mounted() {
    await this.model.ready()
    await this.$nextTick()
    this.modelLayersInfo = this.model.modelLayersInfo
  },

  beforeDestroy() {
    this.model.cleanup()
    this.model.events.removeAllListeners()
  },

  data() {
    const visualizationSelect = this.visualizations[0]
    const visualizationSelectList = [{ text: 'None', value: 'None' }]
    if (['squeezenet_v1.1', 'inception_v3', 'densenet121'].includes(this.modelName)) {
      visualizationSelectList.push({ text: 'Class Activation Mapping', value: 'CAM' })
    }

    return {
      useGPU: this.hasWebGL,
      modelLoading: true,
      modelLoadingProgress: 0,
      modelInitializing: true,
      modelInitProgress: 0,
      modelLayersInfo: [],
      modelRunning: false,
      inferenceTime: null,
      imageURLInput: '',
      imageURLSelect: null,
      imageURLSelectList: IMAGE_URLS,
      imageLoading: false,
      imageLoadingError: false,
      visualizationSelect,
      visualizationSelectList,
      colormapSelect: 'transparency',
      colormapSelectList: COLORMAPS,
      colormapAlpha: 0.7,
      showVis: false,
      output: null
    }
  },

  computed: {
    outputClasses() {
      if (!this.output) {
        const empty = []
        for (let i = 0; i < 5; i++) {
          empty.push({ name: '-', probability: 0 })
        }
        return empty
      }
      return imagenetUtils.imagenetClassesTopK(this.output, 5)
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
        this.updateVis(this.outputClasses[0].index)
      }
    },
    colormapSelect() {
      this.updateVis(this.outputClasses[0].index)
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
              }, 10)
            })
          }
        },
        {
          maxWidth: this.imageSize,
          maxHeight: this.imageSize,
          cover: true,
          crop: true,
          canvas: true,
          crossOrigin: 'Anonymous'
        }
      )
    },
    runModel() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)

      const preprocessedData = this.preprocess(imageData)

      const inputName = this.model.inputLayerNames[0]
      const outputName = this.model.outputLayerNames[0]
      const inputData = { [inputName]: preprocessedData }
      this.model.predict(inputData).then(outputData => {
        this.inferenceTime = this.model.predictStats.forwardPass
        this.output = outputData[outputName]
        this.modelRunning = false
        this.updateVis(this.outputClasses[0].index)
      })
    },
    updateVis(index) {
      if (!this.output || !this.model.visMap.has(this.visualizationSelect)) return

      const vis = this.model.visMap.get(this.visualizationSelect)
      const height = this.imageSize
      const width = this.imageSize
      const visDataArr = ndarray(vis.data, vis.shape)
      const visClassDataArr = ndarray(new Float32Array(vis.shape[0] * vis.shape[1]), [vis.shape[0], vis.shape[1]])
      ops.assign(visClassDataArr, visDataArr.pick(null, null, index))
      const visClassDataArrResized = ndarray(new Float32Array(height * width), [height, width])
      resample(visClassDataArrResized, visClassDataArr)

      const imageDataArr = ndarray(new Uint8ClampedArray(height * width * 4), [height, width, 4])
      if (this.colormapSelect === 'transparency') {
        const alpha = visClassDataArrResized
        ops.mulseq(alpha, -255)
        ops.addseq(alpha, 255)
        ops.assign(imageDataArr.pick(null, null, 3), alpha)
      } else {
        const colormap = this.colormapSelect
        for (let i = 0, len = visClassDataArrResized.data.length; i < len; i++) {
          const rgb = colormap(visClassDataArrResized.data[i]).rgb()
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

      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    }
  }
}
</script>

<style lang="postcss" scoped>
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
  position: relative;

  & .colormap-alpha {
    position: relative;

    & label {
      position: absolute;
      color: var(--color-darkgray);
      font-size: 10px;
    }
  }

  & .visualization-instruction {
    position: absolute;
    bottom: 10px;
    left: 0;
    font-size: 12px;
    color: var(--color-lightgray);
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

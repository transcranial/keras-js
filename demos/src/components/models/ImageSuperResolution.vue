<template>
  <div class="demo">
    <v-alert outline color="error" icon="priority_high" :value="!hasWebGL">
      Note: this browser does not support WebGL 2 or the features necessary to run in GPU mode.
    </v-alert>
    <div class="ui-container">
      <v-layout row justify-center align-center class="mb-2">
        <v-flex xs5 md3 class="model-select">
          <div v-if="modelLoading || modelInitializing" class="loading-indicator">
            <v-progress-circular indeterminate color="primary" />
          </div>
          <v-select
            v-model="modelSelect"
            :disabled="modelLoading || modelInitializing"
            :items="modelSelectList"
            label="Super-Resolution Model"
            max-height="750"
          ></v-select>
        </v-flex>
      </v-layout>
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
      <div class="input-image-panel text-xs-center pa-2 my-4">
        <div class="subheading text-xs-left mb-2">Input Image</div>
        <div class="input-canvas-container">
          <canvas id="input-canvas" class="white" />
          <div class="message">
            <div v-if="imageLoading" class="loading-indicator subheading orange--text">Loading...</div>
            <div v-else-if="imageLoadingError" class="error-message subheading error--text">Error loading URL</div>
          </div>
        </div>
      </div>
      <div class="output-image-panel text-xs-center pa-2 my-4">
        <div class="subheading text-xs-left mb-2">Super-Resolution Image (2X)</div>
        <image-comparison :height="outputImageShape[0]" :width="outputImageShape[1]">
          <canvas style="background:white;" slot="after" id="output-canvas" />
          <canvas style="background:white;" slot="before" id="scaled-input-canvas" />
          <div v-if="modelRunning" slot="message" class="loading-indicator display-1 orange--text">Running...</div>
          <div v-else-if="modelRunningError" slot="message" class="error-message display-1 error--text">Error running neural network model</div>
          <span v-if="inputImageShape[0] > 0 && inputImageShape[1] > 0" slot="beforeLabel">Bilinear Upsampling</span>
          <span v-if="inputImageShape[0] > 0 && inputImageShape[1] > 0" slot="afterLabel">Super-Resolution Output</span>
        </image-comparison>
      </div>
    </div>
    <architecture-diagram :modelLayersInfo="modelLayersInfo"></architecture-diagram>
  </div>
</template>

<script>
import _ from 'lodash'
import Promise from 'bluebird'
import loadImage from 'blueimp-load-image'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import { IMAGE_URLS } from '../../data/sample-image-urls'
import ImageComparison from '../common/ImageComparison'
import ArchitectureDiagram from '../common/ArchitectureDiagram'

const MODEL_SELECT_LIST = [
  { text: 'Distilled ResNet SR', value: 'drnsr' },
  { text: 'SR CNN', value: 'sr' },
  { text: 'Expanded SR CNN', value: 'esr' },
  { text: 'ResNet SR', value: 'rnsr' },
  { text: 'Deep Denoising Autoencoder SR CNN', value: 'ddsr' }
]

const DEV_FILEPATH_PREFIX = '/demos/data/image_super_resolution/'
const PROD_FILEPATH_PREFIX = 'https://transcranial.github.io/keras-js-demos-data/image_super_resolution/'

const DEFAULT_MODEL = MODEL_SELECT_LIST[0].value
const DEFAULT_FILEPATH =
  process.env.NODE_ENV === 'production'
    ? `${PROD_FILEPATH_PREFIX}${DEFAULT_MODEL}.bin`
    : `${DEV_FILEPATH_PREFIX}${DEFAULT_MODEL}.bin`

export default {
  props: ['hasWebGL'],

  components: { ImageComparison, ArchitectureDiagram },

  created() {
    // store module on component instance as non-reactive object
    this.model = new KerasJS.Model({
      filepath: DEFAULT_FILEPATH,
      gpu: this.hasWebGL
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
    return {
      useGPU: this.hasWebGL,
      modelSelect: DEFAULT_MODEL,
      modelSelectList: MODEL_SELECT_LIST,
      modelLoading: true,
      modelLoadingProgress: 0,
      modelInitializing: true,
      modelInitProgress: 0,
      modelLayersInfo: [],
      modelRunning: false,
      modelRunningError: false,
      imageURLInput: '',
      imageURLSelect: null,
      imageURLSelectList: IMAGE_URLS,
      imageLoading: false,
      imageLoadingError: false,
      srcMaxWidth: 400,
      patchSize: 32,
      patchStride: 28,
      inputImageShape: [0, 0],
      outputImageShape: [0, 0],
      output: null
    }
  },

  computed: {
    trueUpscaling() {
      return ['rnsr', 'drnsr'].includes(this.modelSelect)
    }
  },

  watch: {
    async modelSelect(newVal) {
      this.clearAll()
      this.model.cleanup()
      this.model.events.removeAllListeners()
      this.modelLoading = true
      this.modelLoadingProgress = 0
      this.modelInitializing = true
      this.modelInitProgress = 0
      this.modelLayersInfo = []
      this.model = new KerasJS.Model({
        filepath:
          process.env.NODE_ENV === 'production'
            ? `${PROD_FILEPATH_PREFIX}${newVal}.bin`
            : `${DEV_FILEPATH_PREFIX}${newVal}.bin`,
        gpu: this.hasWebGL
      })
      this.model.events.on('loadingProgress', this.handleLoadingProgress)
      this.model.events.on('initProgress', this.handleInitProgress)
      await this.model.ready()
      await this.$nextTick()
      this.modelLayersInfo = this.model.modelLayersInfo
    },
    imageURLSelect(newVal) {
      this.imageURLInput = newVal
      this.loadImageToCanvas(newVal)
    },
    useGPU(newVal) {
      this.model.toggleGPU(newVal)
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
            // adjust canvas dimensions
            const inputCanvas = document.getElementById('input-canvas')
            inputCanvas.width = img.width
            inputCanvas.height = img.height
            this.inputImageShape = [img.height, img.width]
            const outputCanvas = document.getElementById('output-canvas')
            outputCanvas.width = img.width * 2
            outputCanvas.height = img.height * 2
            this.outputImageShape = [img.height * 2, img.width * 2]
            const scaledInputCanvas = document.getElementById('scaled-input-canvas')
            scaledInputCanvas.width = img.width * 2
            scaledInputCanvas.height = img.height * 2

            // load image data onto input canvas
            inputCanvas.getContext('2d').drawImage(img, 0, 0)
            this.imageLoadingError = false
            this.imageLoading = false

            // bilinear scale input image to compare to
            const scaledImg = loadImage.scale(img, { minWidth: img.width * 2, maxWidth: img.width * 2 })
            scaledInputCanvas.getContext('2d').drawImage(scaledImg, 0, 0)

            // run model
            this.modelRunning = true
            this.modelRunningError = false
            this.$nextTick(function() {
              setTimeout(() => {
                this.runModel()
              }, 10)
            })
          }
        },
        { maxWidth: this.srcMaxWidth, canvas: true, crossOrigin: 'Anonymous' }
      )
    },
    generatePatches() {
      const ctx = this.trueUpscaling
        ? document.getElementById('input-canvas').getContext('2d')
        : document.getElementById('scaled-input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData
      const dataArr = ndarray(new Float32Array(data), [height, width, 4])

      const i0List = _.range(0, height, this.patchStride)
      const j0List = _.range(0, width, this.patchStride)

      const patches = []
      const size = this.patchSize
      for (let i0Index = 0; i0Index < i0List.length; i0Index++) {
        const i0 = i0List[i0Index]
        for (let j0Index = 0; j0Index < j0List.length; j0Index++) {
          const j0 = j0List[j0Index]
          const patch = ndarray(new Float32Array(size * size * 3), [size, size, 3])
          const i1 = i0 + size > height ? height : i0 + size
          const j1 = j0 + size > width ? width : j0 + size
          ops.assign(patch.hi(i1 - i0, j1 - j0, 3).lo(0, 0, 0), dataArr.hi(i1, j1, 3).lo(i0, j0, 0))
          ops.divseq(patch, 255)
          patches.push(patch)
        }
      }

      return patches
    },
    combinePatches(patches) {
      const size = this.trueUpscaling ? this.patchSize * 2 : this.patchSize
      const stride = this.trueUpscaling ? this.patchStride * 2 : this.patchStride
      const i0List = _.range(0, this.outputImageShape[0], stride)
      const j0List = _.range(0, this.outputImageShape[1], stride)
      const combinedShape = [this.outputImageShape[0], this.outputImageShape[1], 4]
      const combined = ndarray(new Float32Array(combinedShape.reduce((a, b) => a * b, 1)), combinedShape)
      const overlap = ndarray(new Float32Array(this.outputImageShape.reduce((a, b) => a * b, 1)), this.outputImageShape)
      for (let n = 0, len = patches.length; n < len; n++) {
        const i0 = i0List[Math.floor(n / j0List.length)]
        const j0 = j0List[n % j0List.length]
        const patchHeight = i0 + size > this.outputImageShape[0] ? this.outputImageShape[0] - i0 : size
        const patchWidth = j0 + size > this.outputImageShape[1] ? this.outputImageShape[1] - j0 : size
        ops.addseq(overlap.hi(i0 + patchHeight, j0 + patchWidth).lo(i0, j0), 1)
        ops.addeq(
          combined.hi(i0 + patchHeight, j0 + patchWidth, 3).lo(i0, j0, 0),
          patches[n].hi(patchHeight, patchWidth, 3).lo(0, 0, 0)
        )
      }

      ops.maxseq(overlap, 1)
      for (let c = 0; c < 3; c++) {
        ops.diveq(combined.pick(null, null, c), overlap)
      }
      ops.mulseq(combined, 255)
      ops.minseq(ops.maxseq(combined, 0), 255)
      // assign alpha channel
      ops.assigns(combined.pick(null, null, 3), 255)

      return combined
    },
    async runModel() {
      // generate preprocessed patches
      const inputPatches = this.generatePatches()
      if (!inputPatches.length) {
        this.modelRunning = false
        return
      }

      const inputName = this.model.inputLayerNames[0]
      const outputName = this.model.outputLayerNames[0]
      const size = this.trueUpscaling ? this.patchSize * 2 : this.patchSize
      try {
        const outputPatches = await Promise.mapSeries(inputPatches, async patch => {
          const inputData = { [inputName]: patch.data }
          const outputData = await this.model.predict(inputData)
          return ndarray(outputData[outputName], [size, size, 3])
        })
        // combine output patches and draw image
        this.output = this.combinePatches(outputPatches)
        this.drawImage()
      } catch (err) {
        console.log(err)
        this.modelRunning = false
        this.modelRunningError = true
        return
      }

      this.modelRunning = false
    },
    drawImage() {
      const image = new ImageData(
        new Uint8ClampedArray(this.output.data),
        this.outputImageShape[1],
        this.outputImageShape[0]
      )
      const ctx = document.getElementById('output-canvas').getContext('2d')
      ctx.putImageData(image, 0, 0)
    },
    clearAll() {
      this.modelRunning = false
      this.modelRunningError = false
      this.imageURLInput = ''
      this.imageURLSelect = null
      this.imageLoading = false
      this.imageLoadingError = false
      this.inputImageShape = [0, 0]
      this.outputImageShape = [0, 0]
      this.output = null

      const canvasIds = ['input-canvas', 'scaled-input-canvas', 'output-canvas']
      canvasIds.forEach(id => {
        const ctx = document.getElementById(id).getContext('2d')
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      })
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

.model-select {
  position: relative;

  & .loading-indicator {
    position: absolute;
    top: 5px;
    left: -50px;
  }
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

.input-image-panel,
.output-image-panel {
  position: relative;
  border: 1px solid var(--color-green-light);
  border-radius: 5px;
}

.input-image-panel {
  & .input-canvas-container {
    display: inline-flex;
    position: relative;

    & .message {
      position: absolute;
      top: 5px;
      right: 5px;
    }
  }
}
</style>

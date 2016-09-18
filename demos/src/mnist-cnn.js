/* global Vue */
import debounce from 'lodash/debounce'

const getMidpoint = (p1, p2) => {
  const [x1, y1] = p1
  const [x2, y2] = p2
  return [
    x1 + (x2 - x1) / 2,
    y1 + (y2 - y1) / 2
  ]
}

const getCoordinates = e => {
  let { clientX, clientY } = e
  // for touch event
  if (e.touches && e.touches.length) {
    clientX = e.touches[0].clientX
    clientY = e.touches[0].clientY
  }
  const { left, top } = e.target.getBoundingClientRect()
  const [x, y] = [clientX - left, clientY - top]
  return [x, y]
}

export const MnistCnn = Vue.extend({
  template: `
  <div class="demo mnist-cnn">
    <div class="title">Basic Convnet - MNIST</div>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>
    <div class="input-container">
      <div class="input-label">Draw any digit (0-9) here <span class="arrow">⤸</span></div>
      <div class="canvas-container">
        <canvas
          id="input-canvas" width="240" height="240"
          v-on:mousedown="activateDraw"
          v-on:mouseup="deactivateDraw"
          v-on:mousemove="draw"
          v-on:touchstart="activateDraw"
          v-on:touchend="deactivateDraw"
          v-on:touchmove="draw"
        ></canvas>
        <canvas id="input-canvas-scaled" width="28" height="28" style="display: none;"></canvas>
      </div>
      <div class="input-clear" v-on:click="clear">
        <i class="material-icons">clear</i>CLEAR
      </div>
    </div>
    <div class="columns">
      <div class="column">
      </div>
      <div class="column">
      </div>
    </div>
  </div>
  `,

  data: function () {
    return {
      model: new KerasJS.Model({
        model: '/demos/mnist_cnn/mnist_cnn.json',
        weights: '/demos/mnist_cnn/mnist_cnn_weights.buf',
        metadata: '/demos/mnist_cnn/mnist_cnn_metadata.json'
      }),
      modelLoading: true,
      inputData: {
        'input': new Float32Array(784)
      },
      drawing: false,
      strokes: []
    }
  },

  computed: {
    loadingProgress: function () {
      return this.model.getLoadingProgress()
    }
  },

  created: function () {
    // initialize KerasJS model
    this.model.initialize()
    this.model.ready().then(() => {
      this.modelLoading = false
    })
  },

  ready: function () {
    // initialize scaling helper canvas
    const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
    ctxScaled.scale(28 / 240, 28 / 240)
  },

  methods: {
    clear: function (e) {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      this.drawing = false
      this.strokes = []
    },
    activateDraw: function (e) {
      this.drawing = true
      this.strokes.push([])
      let points = this.strokes[this.strokes.length - 1]
      points.push(getCoordinates(e))
    },
    deactivateDraw: function (e) {
      this.drawing = false
      this.processCanvasData()
      this.model.predict(this.inputData)
    },
    draw: function (e) {
      if (!this.drawing) return

      const ctx = document.getElementById('input-canvas').getContext('2d')

      ctx.lineWidth = 12
      ctx.lineJoin = ctx.lineCap = 'round'
      ctx.strokeStyle = '#393E46'

      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

      let points = this.strokes[this.strokes.length - 1]
      points.push(getCoordinates(e))

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
          ctx.quadraticCurveTo(...p1, ...getMidpoint(p1, p2))
          p1 = points[i]
          p2 = points[i + 1]
        }
        ctx.lineTo(...p1)
        ctx.stroke()
      }
    },
    processCanvasData: function () {
      const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
      ctxScaled.drawImage(document.getElementById('input-canvas'), 0, 0)
      const imageDataScaled = ctxScaled.getImageData(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      const { data } = imageDataScaled
      for (let i = 0, len = data.length; i < len; i += 4) {
        this.inputData['input'][i / 4] = data[i + 3] / 255
      }
    }
  }
})

/* global Vue */
import debounce from 'lodash/debounce'
import range from 'lodash/range'

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
    <div class="columns">
      <div class="column">
        <div class="input-container">
          <div class="input-label">Draw any digit (0-9) here <span class="arrow">⤸</span></div>
          <div class="canvas-container">
            <canvas
              id="input-canvas" width="240" height="240"
              @mousedown="activateDraw"
              @mouseup="deactivateDraw"
              @mousemove="draw"
              @touchstart="activateDraw"
              @touchend="deactivateDraw"
              @touchmove="draw"
            ></canvas>
            <canvas id="input-canvas-scaled" width="28" height="28" style="display: none;"></canvas>
          </div>
          <div class="input-clear" v-on:click="clear">
            <i class="material-icons">clear</i>CLEAR
          </div>
        </div>
      </div>
      <div class="column">
        <div class="output">
          <div class="output-class"
            v-bind:class="{ 'predicted': i === predictedClass }"
            v-for="i in outputClasses"
          >
            <div class="output-label">{{ i }}</div>
            <div class="output-bar"
              style="height: {{ Math.round(100 * output[i]) }}px; background: rgba(27, 188, 155, {{ output[i].toFixed(2) }});"
            ></div>
          </div>
        </div>
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
      input: new Float32Array(784),
      output: new Float32Array(10),
      outputClasses: range(10),
      drawing: false,
      strokes: []
    }
  },

  computed: {
    loadingProgress: function () {
      return this.model.getLoadingProgress()
    },
    predictedClass: function () {
      if (this.output.reduce((a, b) => a + b, 0) === 0) {
        return -1
      }
      return this.output.reduce((argmax, n, i) => n > this.output[argmax] ? i : argmax, 0)
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
      const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
      ctxScaled.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      this.output = new Float32Array(10)
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
      this.output = this.model.predict({ input: this.input }).output
    },
    draw: function (e) {
      if (!this.drawing) return

      const ctx = document.getElementById('input-canvas').getContext('2d')

      ctx.lineWidth = 15
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
    processCanvasData: debounce(function () {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
      ctxScaled.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      ctxScaled.drawImage(document.getElementById('input-canvas'), 0, 0)
      const imageDataScaled = ctxScaled.getImageData(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      const { data } = imageDataScaled
      this.input = new Float32Array(784)
      for (let i = 0, len = data.length; i < len; i += 4) {
        this.input[i / 4] = data[i + 3] / 255
      }
    }, 200, { leading: true, trailing: true })
  }
})

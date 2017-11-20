import Tensor from '../Tensor'
import { webgl2 } from '../WebGL2'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import resample from 'ndarray-resample'
import programSource from './CAM.glsl'

/**
 * Class Activation Maps
 */
export default class CAM {
  /**
   * @param {Object} [attrs] - visualization layer attributes
   */
  constructor(attrs = {}) {
    this.modelLayersMap = attrs.modelLayersMap
    this.gpu = attrs.gpu

    if (!this.modelLayersMap) {
      throw new Error(`[CAM] modelLayersMap is required`)
    }

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(programSource)
    }
  }

  /**
   * Checks whether CAM can be computed directly (requires GlobalAveragePooling2D layer)
   * Grad-CAM generalizes this to arbitrary architectures, and may be implemented in the future.
   *
   * @param {number} width
   * @param {number} height
   */
  initialize(width, height) {
    this.modelLayersMap.forEach(layer => {
      if (layer.layerClass === 'GlobalAveragePooling2D') {
        this.enabled = true
        this.poolLayer = layer
      }
    })

    if (this.enabled && !this.data) {
      this.width = width
      this.height = height
      this.data = new Float32Array(this.width * this.height)
    }
  }

  /**
   * Update visualization output
   */
  update() {
    if (!this.enabled) return

    // GlobalAveragePooling2D layer provides feature map weights
    this.weights = this.poolLayer.output
    // get feature maps from preceding layer
    this.featureMaps = this.modelLayersMap.get(this.poolLayer.inbound[0]).output

    if (this.gpu) {
      this._updateGPU()
    } else {
      this._updateCPU()
    }

    const dataArr = ndarray(this.data, [this.height, this.width])
    resample(dataArr, this.output.tensor)
  }

  _updateCPU() {
    this.inputShape = this.featureMaps.tensor.shape
    this.outputShape = this.inputShape.slice(0, 2)
    if (!this.output) {
      this.output = new Tensor([], this.outputShape)
    }

    const channels = this.weights.tensor.shape[0]
    const weightedFeatureMap = new Tensor([], this.outputShape)
    ops.assigns(this.output.tensor, 0)
    for (let c = 0; c < channels; c++) {
      ops.muls(weightedFeatureMap.tensor, this.featureMaps.tensor.pick(null, null, c), this.weights.tensor.get(c))
      ops.addeq(this.output.tensor, weightedFeatureMap.tensor)
    }
    ops.divseq(this.output.tensor, ops.sum(this.weights.tensor))
    ops.maxseq(this.output.tensor, 0)

    // normalize 0-1
    ops.divseq(this.output.tensor, ops.sup(this.output.tensor))
  }

  _updateGPU() {
    if (this.featureMaps.is2DReshaped) {
      this.inputShape = this.featureMaps.originalShape
    } else {
      this.inputShape = this.featureMaps.tensor.shape
    }

    this.outputShape = this.inputShape.slice(0, 2)
    if (!this.output) {
      this.output = new Tensor([], this.outputShape)
      this.output.createGLTexture({ type: '2d', format: 'float' })
    }

    webgl2.runProgram({
      program: this.program,
      output: this.output,
      inputs: [{ input: this.featureMaps, name: 'featureMaps' }, { input: this.weights, name: 'weights' }],
      uniforms: [
        { value: this.output.glTextureShape[0], type: 'int', name: 'rows' },
        { value: this.output.glTextureShape[1], type: 'int', name: 'cols' }
      ]
    })

    // GPU -> CPU data transfer
    this.output.transferFromGLTexture()

    // normalize 0-1
    ops.divseq(this.output.tensor, ops.sup(this.output.tensor))
  }
}

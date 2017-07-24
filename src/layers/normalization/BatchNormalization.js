import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../activations'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'
import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'

/**
 * BatchNormalization layer class
 */
export default class BatchNormalization extends Layer {
  /**
   * Creates an BatchNormalization layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'BatchNormalization'

    const { epsilon = 0.001, axis = -1, center = true, scale = true } = attrs

    this.epsilon = epsilon
    this.center = center
    this.scale = scale

    // no batch axis, so axis is less 1 compared to representation in keras
    // will be set in call(), as input tensor shape is needed to calculate axis
    // if axis < 0
    this.axis = axis
    this.axisNormalized = false

    // Layer weights specification
    this.params = []
    if (this.scale) {
      this.params.push('gamma')
    }
    if (this.center) {
      this.params.push('beta')
    }
    this.params = this.params.concat(['moving_mean', 'moving_variance'])

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./BatchNormalization.webgl2.glsl'))
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (!this.axisNormalized) {
      this.axis = this.axis < 0 ? x.tensor.shape.length + this.axis : this.axis - 1
      this.axisNormalized = true
    }

    if (this.gpu) {
      this._call_gpu(x)
    } else {
      this._call_cpu(x)
    }
    return this.output
  }

  /**
   * CPU call
   */
  _call_cpu(x) {
    let broadcast = []
    for (let d = 0; d < x.tensor.shape.length; d++) {
      if (d === this.axis) broadcast.push(1)
      else broadcast.push(null)
    }

    // broadcast weights
    let _gamma = new Tensor([], x.tensor.shape)
    let _beta = new Tensor([], x.tensor.shape)
    for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
      broadcast[this.axis] = i
      if (this.scale) {
        ops.assigns(_gamma.tensor.pick(...broadcast), this.weights.gamma.tensor.get(i))
      }
      if (this.center) {
        ops.assigns(_beta.tensor.pick(...broadcast), this.weights.beta.tensor.get(i))
      }
    }

    let _mean = new Tensor([], x.tensor.shape)
    let _std = new Tensor([], x.tensor.shape)

    // feature-wise normalization
    for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
      broadcast[this.axis] = i
      ops.assigns(_mean.tensor.pick(...broadcast), this.weights.moving_mean.tensor.get(i))
      ops.assigns(_std.tensor.pick(...broadcast), this.weights.moving_variance.tensor.get(i) + this.epsilon)
    }
    ops.sqrteq(_std.tensor)

    this.output = new Tensor(x.tensor.data, x.tensor.shape)

    ops.subeq(this.output.tensor, _mean.tensor)
    ops.diveq(this.output.tensor, _std.tensor)
    if (this.scale) {
      ops.muleq(this.output.tensor, _gamma.tensor)
    }
    if (this.center) {
      ops.addeq(this.output.tensor, _beta.tensor)
    }
  }

  /**
   * GPU call
   * Will only work on the 2D-tiled representation for post-convolutional BN
   */
  _call_gpu(x) {
    this.inputShape = x.tensor.shape

    if (x.tensor.shape.length <= 2 && !x.glTexture) {
      x.createGLTexture()
    } else if (x.tensor.shape.length > 2 && !x.glTextureIsTiled) {
      const normAxisLength = x.tensor.shape[this.axis]
      const otherAxes = [...x.tensor.shape.slice(0, this.axis), ...x.tensor.shape.slice(this.axis + 1)]
      const otherAxesSize = otherAxes.reduce((a, b) => a * b, 1)
      const tiled = new Tensor([], [otherAxesSize, normAxisLength])
      const otherAxesData = new Tensor([], otherAxes)
      const otherAxesDataRaveled = new Tensor([], [otherAxesSize])
      const axisSlices = Array(x.tensor.shape.length).fill(null)
      for (let n = 0; n < normAxisLength; n++) {
        axisSlices[this.axis] = n
        ops.assign(otherAxesData.tensor, x.tensor.pick(...axisSlices))
        otherAxesDataRaveled.replaceTensorData(otherAxesData.tensor.data)
        ops.assign(tiled.tensor.pick(null, n), otherAxesDataRaveled.tensor)
      }

      x = tiled
      x.glTextureIsTiled = true
      x.untiledShape = this.inputShape
      x.createGLTexture()
    }

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new Tensor([], x.glTextureShape)
      this.output.createGLTexture()
      if (x.glTextureIsTiled) {
        this.output.glTextureIsTiled = x.glTextureIsTiled
        this.output.untiledShape = x.untiledShape
      }
    }

    webgl2.selectProgram(this.program)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    const textures = [x.glTexture]
    const textureTypes = ['2d']
    const textureNames = ['X']
    if (this.scale) {
      textures.push(this.weights['gamma'].glTexture)
      textureTypes.push('2d')
      textureNames.push('gamma')
    }
    if (this.center) {
      textures.push(this.weights['beta'].glTexture)
      textureTypes.push('2d')
      textureNames.push('beta')
    }
    textures.push(this.weights['moving_mean'].glTexture, this.weights['moving_variance'].glTexture)
    textureTypes.push('2d', '2d')
    textureNames.push('mean', 'std')
    webgl2.bindInputTextures(this.program, textures, textureTypes, textureNames)
    const uniforms = [
      this.epsilon,
      this.output.glTextureShape[0],
      this.output.glTextureShape[1],
      +this.scale,
      +this.center
    ]
    const uniformTypes = ['float', 'int', 'int', 'bool', 'bool']
    const uniformNames = ['epsilon', 'rows', 'cols', 'scale', 'center']
    webgl2.bindUniforms(this.program, uniforms, uniformTypes, uniformNames)
    webgl2.runProgram()

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.tensor.data = webgl2.readData(this.output.glTextureShape)
      if (this.output.glTextureIsTiled) {
        this.output.reshapeTensorFromTiled(this.axis)
      }
    }
  }
}

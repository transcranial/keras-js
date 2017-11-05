import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * _GlobalPooling1D layer class
 */
export default class _GlobalPooling1D extends Layer {
  /**
   * Creates a _GlobalPooling1D layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = '_GlobalPooling1D'

    const { data_format = 'channels_last' } = attrs
    this.dataFormat = data_format

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max'

    // GPU setup
    if (this.gpu) {
      this.poolingProgram = webgl2.compileProgram(require('./_GlobalPooling.glsl'))
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.gpu) {
      this._callGPU(x)
    } else {
      this._callCPU(x)
    }
    return this.output
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    const [steps, features] = x.tensor.shape
    this.output = new Tensor([], [features])
    for (let i = 0, len = features; i < len; i++) {
      if (this.poolingFunc === 'max') {
        this.output.tensor.set(i, ops.sup(x.tensor.pick(null, i)))
      } else if (this.poolingFunc === 'average') {
        this.output.tensor.set(i, ops.sum(x.tensor.pick(null, i)) / steps)
      }
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture) {
      x.createGLTexture()
    }
    this.inputShape = x.tensor.shape

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new Tensor([], [this.inputShape[1]])
      this.output.createGLTexture()
    }

    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max'

    webgl2.selectProgram(this.poolingProgram)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    const uniforms = [this.inputShape[0], +isMaxPooling]
    const uniformTypes = ['int', 'bool']
    const uniformNames = ['channelDataSize', 'isMaxPooling']
    webgl2.bindUniforms(this.poolingProgram, uniforms, uniformTypes, uniformNames)
    const textures = [x.glTexture]
    const textureTypes = ['2d']
    const textureNames = ['x']
    webgl2.bindInputTextures(this.poolingProgram, textures, textureTypes, textureNames)
    webgl2.runProgram()

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
    }
  }
}

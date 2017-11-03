import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * _GlobalPooling3D layer class
 */
export default class _GlobalPooling3D extends Layer {
  /**
   * Creates a _GlobalPooling3D layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = '_GlobalPooling3D'

    const { data_format = 'channels_last' } = attrs
    this.dataFormat = data_format

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max'

    // GPU setup
    if (this.gpu) {
      this.poolingProgram = webgl2.compileProgram(require('./_GlobalPooling.webgl2.glsl'))
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
    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 3, 0)
    }

    const [dim1, dim2, dim3, channels] = x.tensor.shape
    this.output = new Tensor([], [channels])
    for (let i = 0, len = channels; i < len; i++) {
      if (this.poolingFunc === 'max') {
        this.output.tensor.set(i, ops.sup(x.tensor.pick(null, null, null, i)))
      } else if (this.poolingFunc === 'average') {
        this.output.tensor.set(i, ops.sum(x.tensor.pick(null, null, null, i)) / (dim1 * dim2 * dim3))
      }
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (x.glTextureIsTiled) {
      this.inputShape = x.untiledShape
    } else {
      // convert to channels_last ordering
      if (this.dataFormat === 'channels_first') {
        x.tensor = x.tensor.transpose(1, 2, 3, 0)
      }
      this.inputShape = x.tensor.shape
      x.reshapeTensorToTiled()
      x.createGLTexture()
    }

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new Tensor([], [this.inputShape[3]])
      this.output.createGLTexture()
    }

    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max'

    webgl2.selectProgram(this.poolingProgram)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    const uniforms = [this.inputShape[0] * this.inputShape[1] * this.inputShape[2], +isMaxPooling]
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

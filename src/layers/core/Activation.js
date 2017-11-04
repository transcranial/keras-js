import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import * as activations from '../../activations'

/**
 * Activation layer class
 */
export default class Activation extends Layer {
  /**
   * Creates an Activation layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {String} [attrs.activation] - name of activation function
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Activation'

    const { activation = 'linear' } = attrs

    this.activation = activation
    this.activationFunc = activations[activation]

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require(`../../activations/${this.activation}.webgl2.glsl`))
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.activation === 'linear') {
      this.output = x
      return this.output
    }

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
    this.output = x
    this.activationFunc(this.output)
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

    if (!this.output) {
      this.output = new Tensor([], x.glTextureShape)
      this.output.createGLTexture()
      if (x.is2DReshaped) {
        this.output.is2DReshaped = x.is2DReshaped
        this.output.originalShape = x.originalShape
      }
    }

    webgl2.selectProgram(this.program)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    const textures = [x.glTexture]
    const textureTypes = ['2d']
    const textureNames = ['x']
    webgl2.bindInputTextures(this.program, textures, textureTypes, textureNames)
    webgl2.runProgram()

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
      if (this.output.is2DReshaped) {
        this.output.reshapeFrom2D()
      }
    }
  }
}

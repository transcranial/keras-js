import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import cwise from 'cwise'

/**
 * ThresholdedReLU advanced activation layer class
 */
export default class ThresholdedReLU extends Layer {
  /**
   * Creates a ThresholdedReLU activation layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.theta] - float >= 0. Threshold location of activation.
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'ThresholdedReLU'

    const { theta = 1 } = attrs

    this.theta = theta

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./ThresholdedReLU.glsl'))
    }
  }

  _compute = cwise({
    args: ['array', 'scalar'],
    body: function(_x, theta) {
      _x = _x * Number(_x > theta)
    }
  })

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
    this.output = x
    this._compute(this.output.tensor, this.theta)
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
      if (x.is1D) {
        this.output.is1D = x.is1D
      } else if (x.is2DReshaped) {
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
    const uniforms = [this.theta]
    const uniformTypes = ['float']
    const uniformNames = ['theta']
    webgl2.bindUniforms(this.program, uniforms, uniformTypes, uniformNames)
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

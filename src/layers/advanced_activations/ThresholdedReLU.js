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
   * @param {number} attrs.theta - float >= 0. Threshold location of activation.
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'ThresholdedReLU'

    const { theta = 1 } = attrs

    this.theta = theta

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./ThresholdedReLU.webgl2.glsl'))
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
    this.output = x
    this._compute(this.output.tensor, this.theta)
  }

  /**
   * GPU call
   */
  _call_gpu(x) {
    if (!x.glTexture) {
      x.createGLTexture()
    }

    this.output = this.output || new Tensor([], x.tensor.shape)
    if (!this.output.glTexture) {
      this.output.createGLTexture()
    }

    webgl2.selectProgram(this.program)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    webgl2.bindInputTextures(this.program, [x.glTexture], ['x'])
    webgl2.bindUniforms(this.program, [this.theta], ['float'], ['theta'])
    webgl2.runProgram()

    if (this.outbound.length === 0) {
      this.output.tensor.data = webgl2.readData(this.output.glTextureShape)
    }
  }
}

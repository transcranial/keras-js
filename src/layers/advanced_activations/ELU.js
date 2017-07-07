import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import cwise from 'cwise'

/**
 * ELU advanced activation layer class
 */
export default class ELU extends Layer {
  /**
   * Creates a ELU activation layer
   * @param {number} attrs.alpha - scale for the negative factor
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'ELU'

    const { alpha = 1.0 } = attrs

    this.alpha = alpha

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./ELU.webgl2.glsl'))
    }
  }

  _compute = cwise({
    args: ['array', 'scalar'],
    body: function(_x, alpha) {
      _x = Math.max(_x, 0) + alpha * (Math.exp(Math.min(_x, 0)) - 1)
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
    this._compute(this.output.tensor, this.alpha)
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
    webgl2.bindUniforms(this.program, [this.alpha], ['float'], ['alpha'])
    webgl2.runProgram()

    if (this.outbound.length === 0) {
      this.output.tensor.data = webgl2.readData(this.output.glTextureShape)
    }
  }
}

import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import cwise from 'cwise'

/**
 * PReLU advanced activation layer class
 *
 * reference code:
 * ```
 * pos = K.relu(x)
 * neg = self.alpha * (x - abs(x)) * 0.5
 * return pos + neg
 * ```
 */
export default class PReLU extends Layer {
  /**
   * Creates a PReLU activation layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'PReLU'

    // Layer weights specification
    this.params = ['alpha']

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./PReLU.glsl'))
    }
  }

  _compute = cwise({
    args: ['array', 'array'],
    body: function(_x, alpha) {
      _x = Math.max(_x, 0) + alpha * Math.min(_x, 0)
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
    this._compute(this.output.tensor, this.weights['alpha'].tensor)
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
        this.output.indicesForReshaped = x.indicesForReshaped
      }
    }

    webgl2.selectProgram(this.program)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    const textures = [x.glTexture, this.weights['alpha'].glTexture]
    const textureTypes = ['2d', '2d']
    const textureNames = ['x', 'alpha']
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

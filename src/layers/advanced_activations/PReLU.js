import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import cwise from 'cwise'

/**
 * PReLU advanced activation layer class
 * reference code:
 * ```
 * pos = K.relu(x)
 * neg = self.alphas * (x - abs(x)) * 0.5
 * return pos + neg
 * ```
 */
export default class PReLU extends Layer {
  /**
   * Creates a PReLU activation layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'PReLU'

    // Layer weights specification
    this.params = ['alphas']

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./PReLU.webgl2.glsl'))
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
    this._compute(this.output.tensor, this.weights.alphas.tensor)
  }

  /**
   * GPU call
   */
  _call_gpu(x) {
    if (!x.glTexture) {
      x.createGLTexture()
    }

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
    const textures = [x.glTexture, this.weights['alphas'].glTexture]
    const textureTypes = ['2d', '2d']
    const textureNames = ['x', 'alphas']
    webgl2.bindInputTextures(this.program, textures, textureTypes, textureNames)
    webgl2.runProgram()

    if (this.outbound.length === 0) {
      this.output.tensor.data = webgl2.readData(this.output.glTextureShape)
    }
  }
}

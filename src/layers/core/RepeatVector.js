import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import unsqueeze from 'ndarray-unsqueeze'
import tile from 'ndarray-tile'

/**
 * RepeatVector layer class
 * Turns 2D tensors of shape [features] to 3D tensors of shape [n, features].
 * Note there is no concept of batch size in these layers (single-batch) so we're actually going from 1D to 2D.
 */
export default class RepeatVector extends Layer {
  /**
   * Creates a RepeatVector layer
   * @param {number} attrs.n
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'RepeatVector'

    const { n = 1 } = attrs
    this.n = n

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./RepeatVector.webgl2.glsl'))
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
    if (x.tensor.shape.length !== 1) {
      throw new Error(`${this.name} [RepeatVector layer] Only 1D tensor inputs allowed.`)
    }
    this.output = new Tensor([], [this.n, x.tensor.shape[1]])
    this.output.tensor = tile(unsqueeze(x.tensor, 0), [this.n, 1])
  }

  /**
   * GPU call
   */
  _call_gpu(x) {
    if (!x.glTexture) {
      x.createGLTexture()
    }

    if (!this.output) {
      this.output = new Tensor([], [this.n, x.glTextureShape[1]])
      this.output.createGLTexture()
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
    }
  }
}

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
    const textureNames = ['x']
    webgl2.bindInputTextures(this.program, textures, textureTypes, textureNames)
    const uniforms = [this.alpha]
    const uniformTypes = ['float']
    const uniformNames = ['alpha']
    webgl2.bindUniforms(this.program, uniforms, uniformTypes, uniformNames)
    webgl2.runProgram()

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
      if (this.output.glTextureIsTiled) {
        this.output.reshapeTensorFromTiled()
      }
    }
  }
}

import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import { relu } from '../../activations'

/**
 * LeakyReLU advanced activation layer class
 */
export default class LeakyReLU extends Layer {
  /**
   * Creates a LeakyReLU activation layer
   * @param {number} attrs.alpha - negative slope coefficient
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'LeakyReLU'

    const { alpha = 0.3 } = attrs

    this.alpha = alpha

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./LeakyReLU.webgl2.glsl'))
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
    this.output = x
    relu(this.output, { alpha: this.alpha })
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

    if (this.outbound.length === 0) {
      this.output.tensor.data = webgl2.readData(this.output.glTextureShape)
    }
  }
}

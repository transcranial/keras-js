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
   * @param {String} attrs.activation - name of activation function
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
    this.activationFunc(this.output)
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
    webgl2.runProgram()

    if (this.outbound.length === 0) {
      this.output.tensor.data = webgl2.readData(this.output.glTextureShape)
    }
  }
}

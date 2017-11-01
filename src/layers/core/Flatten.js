import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'

/**
 * Flatten layer class
 * Turns tensor into 1-d. Note there is no concept of batch size in these layers (single-batch).
 */
export default class Flatten extends Layer {
  /**
   * Creates a Flatten layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Flatten'

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(require('./Flatten.webgl2.glsl'))
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
    if (x.tensor.shape.length <= 1) {
      this.output = x
    } else {
      this.output = new Tensor([], [x.tensor.shape.reduce((a, b) => a * b, 1)])
      this.output.replaceTensorData(x.tensor.data)
    }
  }

  /**
   * GPU call
   */
  _call_gpu(x) {
    if (!x.glTexture) {
      if (x.tensor.shape.length <= 2) {
        x.createGLTexture()
      } else if (x.tensor.shape.length > 2 && !x.glTextureIsTiled) {
        x.reshapeTensorToTiled()
        x.createGLTexture()
      }
    }

    if (!this.output) {
      this.output = new Tensor([], [x.glTextureShape.reduce((a, b) => a * b, 1)])
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

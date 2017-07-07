import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../activations'
import { webgl2 } from '../../WebGL2'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'

/**
 * Dense layer class
 */
export default class Dense extends Layer {
  /**
   * Creates a Dense layer
   * @param {Number} attrs.units - output dimension size
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Dense'

    const { units = 1, activation = 'linear', input_dim = null, use_bias = true } = attrs

    this.activation = activation
    this.activationFunc = activations[this.activation]
    this.units = units
    this.input_dim = input_dim
    this.use_bias = use_bias

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel']

    // Input shape specification
    if (this.input_dim) {
      this.inputShape = [this.input_dim]
    }

    // Output
    this.output_preactiv = new Tensor([], [this.units])
    this.output = new Tensor([], [this.units])

    // GPU setup
    if (this.gpu) {
      this.mainProgram = webgl2.compileProgram(require('./Dense.webgl2.glsl'))
      this.activationProgram = webgl2.compileProgram(require(`../../activations/${this.activation}.webgl2.glsl`))
      this.output_preactiv.createGLTexture()
      this.output.createGLTexture()
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
    if (this.use_bias) {
      ops.assign(this.output.tensor, this.weights['bias'].tensor)
    }
    gemv(1, this.weights['kernel'].tensor.transpose(1, 0), x.tensor, 1, this.output.tensor)
    this.activationFunc(this.output)
  }

  /**
   * GPU call
   */
  _call_gpu(x) {
    if (!x.glTexture) {
      x.createGLTexture()
    }

    webgl2.selectProgram(this.mainProgram)
    webgl2.bindOutputTexture(this.output_preactiv.glTexture, this.output_preactiv.glTextureShape)
    const textures = [x.glTexture, ...this.params.map(p => this.weights[p].glTexture)]
    const textureNames = ['x', ...this.params]
    webgl2.bindInputTextures(this.mainProgram, textures, textureNames)
    const uniforms = [this.use_bias ? 1 : 0, ...this.weights['kernel'].glTextureShape]
    const uniformTypes = ['bool', 'int', 'int']
    const uniformNames = ['use_bias', 'M', 'N']
    webgl2.bindUniforms(this.mainProgram, uniforms, uniformTypes, uniformNames)
    webgl2.runProgram()

    webgl2.selectProgram(this.activationProgram)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    webgl2.bindInputTextures(this.activationProgram, [this.output_preactiv.glTexture], ['x'])
    webgl2.runProgram()

    if (this.outbound.length === 0) {
      this.output.tensor.data = webgl2.readData(this.output.glTextureShape)
    }
  }
}

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
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.units] - output dimension size
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
    this.outputPreactiv = new Tensor([], [this.units])
    this.output = new Tensor([], [this.units])

    // GPU setup
    if (this.gpu) {
      this.matMulProgram = webgl2.compileProgram(require('../../webgl/matMul.glsl'))
      this.activationProgram = webgl2.compileProgram(require(`../../activations/${this.activation}.glsl`))
      this.outputPreactiv.createGLTexture({ type: '2d', format: 'float' })
      this.output.createGLTexture({ type: '2d', format: 'float' })
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
    if (this.use_bias) {
      ops.assign(this.output.tensor, this.weights['bias'].tensor)
    }
    gemv(1, this.weights['kernel'].tensor.transpose(1, 0), x.tensor, 1, this.output.tensor)
    this.activationFunc(this.output)
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture) {
      x.createGLTexture({ type: '2d', format: 'float' })
    }

    // Matrix Multiply
    const matMulInputs = [{ input: x, name: 'A' }, { input: this.weights['kernel'], name: 'B' }]
    if (this.use_bias) {
      matMulInputs.push({ input: this.weights['bias'], name: 'C' })
    }
    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.outputPreactiv,
      inputs: matMulInputs,
      uniforms: [{ value: this.use_bias ? 1 : 0, type: 'bool', name: 'addC' }]
    })

    // Activation
    if (this.activation === 'linear') {
      this.output = this.outputPreactiv
    } else {
      webgl2.runProgram({
        program: this.activationProgram,
        output: this.output,
        inputs: [{ input: this.outputPreactiv, name: 'x' }]
      })
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
    }
  }
}

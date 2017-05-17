import WebGLLayer from '../WebGLLayer'

export default class WebGLActivation extends WebGLLayer {
  constructor() {
    super()
    this.program = this.webgl.createProgram(require('./activation.glsl'))
  }

  static INPUT_TEXTURE_NAME = 'X'
  static RELU_ACTIVATION_UNIFORM_NAME = 'relu'

  /**
   * Bind WebGL input textures.
   *
   * @param {weblas.pipeline.Tensor} input
   */
  _bindInputTextures(input) {
    const gl = this.webgl.context
    this.numTextures = 1
    this._bindInputTexture(this.program, input.texture, gl.TEXTURE0, WebGLActivation.INPUT_TEXTURE_NAME)
  }

  /**
   * Bind WebGL uniforms.
   *
   * @param {string} activation
   */
  _bindUniforms(activation) {
    const gl = this.webgl.context
    if (activation === 'relu') {
      gl.uniform1i(gl.getUniformLocation(this.program, WebGLActivation.RELU_ACTIVATION_UNIFORM_NAME), 1)
    }
  }

  /**
   * Bind WebGL output texture.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} tOut
   */
  _bindOutputTexture(input, tOut) {
    const outputColPad = this.webgl.getPad(input.shape[1])
    this.webgl.bindOutputTexture(input.shape[0], (input.shape[1] + outputColPad) / 4, tOut.texture)
  }

  /**
   * Main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {string} activation
   *
   * @returns {weblas.pipeline.Tensor}
   */
  call(input, activation) {
    this.webgl.selectProgram(this.program)

    // create an empty output Tensor
    const tOut = new weblas.pipeline.Tensor(input.shape, null)

    this._bindInputTextures(input)
    this._bindUniforms(activation)
    this._bindOutputTexture(input, tOut)
    this._compute()
    this._unbindInputTextures()

    return tOut
  }
}

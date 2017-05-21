import WebGLLayer from '../WebGLLayer'

export default class WebGLBatchNorm extends WebGLLayer {
  constructor() {
    super()
    this.program = this.webgl.createProgram(require('./batchnorm.glsl'))
  }

  static INPUT_TEXTURE_NAME = 'X'
  static MEAN_TEXTURE_NAME = 'mean'
  static STD_TEXTURE_NAME = 'std'
  static GAMMA_TEXTURE_NAME = 'gamma'
  static BETA_TEXTURE_NAME = 'beta'
  static EPSILON_UNIFORM_NAME = 'epsilon'
  static OUTPUT_COLS_UNIFORM_NAME = 'outputCols'
  static OUTPUT_COL_PAD_UNIFORM_NAME = 'outputColPad'

  /**
   * Bind WebGL input textures for main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} gamma
   * @param {weblas.pipeline.Tensor} beta
   * @param {weblas.pipeline.Tensor} mean
   * @param {weblas.pipeline.Tensor} std
   */
  _bindInputTextures(input, gamma, beta, mean, std) {
    const gl = this.webgl.context
    this.numTextures = 5
    this._bindInputTexture(this.program, input.texture, gl.TEXTURE0, WebGLBatchNorm.INPUT_TEXTURE_NAME)
    this._bindInputTexture(this.program, mean.texture, gl.TEXTURE1, WebGLBatchNorm.MEAN_TEXTURE_NAME)
    this._bindInputTexture(this.program, std.texture, gl.TEXTURE2, WebGLBatchNorm.STD_TEXTURE_NAME)
    this._bindInputTexture(this.program, gamma.texture, gl.TEXTURE3, WebGLBatchNorm.GAMMA_TEXTURE_NAME)
    this._bindInputTexture(this.program, beta.texture, gl.TEXTURE4, WebGLBatchNorm.BETA_TEXTURE_NAME)
  }

  /**
   * Bind WebGL uniforms for main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {number} epsilon
   */
  _bindUniforms(input, epsilon) {
    const gl = this.webgl.context
    const outputColPad = this.webgl.getPad(input.shape[1])
    gl.uniform1f(gl.getUniformLocation(this.program, WebGLBatchNorm.EPSILON_UNIFORM_NAME), epsilon)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLBatchNorm.OUTPUT_COLS_UNIFORM_NAME), input.shape[1])
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLBatchNorm.OUTPUT_COL_PAD_UNIFORM_NAME), outputColPad)
  }

  /**
   * Bind WebGL output texture for main operation.
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
   * @param {number} epsilon
   * @param {weblas.pipeline.Tensor} gamma
   * @param {weblas.pipeline.Tensor} beta
   * @param {weblas.pipeline.Tensor} [mean]
   * @param {weblas.pipeline.Tensor} [std]
   *
   * @returns {weblas.pipeline.Tensor}
   */
  call(input, epsilon, gamma, beta, mean, std) {
    this.webgl.selectProgram(this.program)

    // create an empty output Tensor
    const tOut = new weblas.pipeline.Tensor(input.shape, null)

    this._bindInputTextures(input, gamma, beta, mean, std)
    this._bindUniforms(input, epsilon)
    this._bindOutputTexture(input, tOut)
    this._compute()
    this._unbindInputTextures()

    return tOut
  }
}

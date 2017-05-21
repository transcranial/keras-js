import WebGLLayer from '../WebGLLayer'

export default class WebGLPooling2D extends WebGLLayer {
  constructor(poolingFunc) {
    super()
    if (poolingFunc === 'max') {
      this.program = this.webgl.createProgram(require('./maxpooling2d.glsl'))
    } else if (poolingFunc === 'average') {
      this.program = this.webgl.createProgram(require('./avgpooling2d.glsl'))
    } else {
      throw new Error(`[WebGLPooling2D] pooling function must be max or average.`)
    }
  }

  static INPUT_TEXTURE_NAME = 'X'
  static POOL_IMAP_TEXTURE_NAME = 'poolIndexMapping'
  static INPUT_ROWS_UNIFORM_NAME = 'inputRows'
  static CHANNELS_UNIFORM_NAME = 'channels'
  static CHANNELS_PAD_UNIFORM_NAME = 'channelsPad'
  static POOL_ELEMENTS_UNIFORM_NAME = 'poolElements'
  static POOL_ELEMENTS_PAD_UNIFORM_NAME = 'poolElementsPad'

  /**
   * Bind WebGL input textures for main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} poolIndexMapping
   */
  _bindInputTextures(input, poolIndexMapping) {
    const gl = this.webgl.context
    this.numTextures = 2
    this._bindInputTexture(this.program, input.texture, gl.TEXTURE0, WebGLPooling2D.INPUT_TEXTURE_NAME)
    this._bindInputTexture(this.program, poolIndexMapping.texture, gl.TEXTURE1, WebGLPooling2D.POOL_IMAP_TEXTURE_NAME)
  }

  /**
   * Bind WebGL uniforms for main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} poolIndexMapping
   */
  _bindUniforms(input, poolIndexMapping) {
    const gl = this.webgl.context
    const nbPatches = input.shape[0]
    const channels = input.shape[1]
    const channelsPad = this.webgl.getPad(channels)
    const poolElements = poolIndexMapping.shape[1]
    const poolElementsPad = this.webgl.getPad(poolElements)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLPooling2D.INPUT_ROWS_UNIFORM_NAME), nbPatches)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLPooling2D.CHANNELS_UNIFORM_NAME), channels)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLPooling2D.CHANNELS_PAD_UNIFORM_NAME), channelsPad)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLPooling2D.POOL_ELEMENTS_UNIFORM_NAME), poolElements)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLPooling2D.POOL_ELEMENTS_PAD_UNIFORM_NAME), poolElementsPad)
  }

  /**
   * Bind WebGL output texture for main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} poolIndexMapping
   * @param {weblas.pipeline.Tensor} tOut
   */
  _bindOutputTexture(input, poolIndexMapping, tOut) {
    const outputLength = poolIndexMapping.shape[0]
    const channels = input.shape[1]
    const outputCols = this.webgl.getPad(channels)
    this.webgl.bindOutputTexture(outputLength, (channels + outputCols) / 4, tOut.texture)
  }

  /**
   * Main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} poolIndexMapping
   *
   * @returns {weblas.pipeline.Tensor}
   */
  call(input, poolIndexMapping) {
    this.webgl.selectProgram(this.program)

    // create an empty output Tensor
    const outputLength = poolIndexMapping.shape[0]
    const channels = input.shape[1]
    const tOut = new weblas.pipeline.Tensor([outputLength, channels], null)

    this._bindInputTextures(input, poolIndexMapping)
    this._bindUniforms(input, poolIndexMapping)
    this._bindOutputTexture(input, poolIndexMapping, tOut)
    this._compute()
    this._unbindInputTextures()

    return tOut
  }
}

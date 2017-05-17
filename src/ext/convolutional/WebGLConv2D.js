import WebGLLayer from '../WebGLLayer'

export default class WebGLConv2D extends WebGLLayer {
  constructor() {
    super()
    this.inputTransformProgram = this.webgl.createProgram(require('./input_transform.glsl'))
    this.mainProgram = this.webgl.createProgram(require('./conv2d.glsl'))
  }

  static INPUT_TEXTURE_NAME = 'X'
  static WEIGHTS_TEXTURE_NAME = 'W'
  static BIAS_TEXTURE_NAME = 'b'
  static INPUT_ROWS_UNIFORM_NAME = 'inputRows'
  static INPUT_COLS_UNIFORM_NAME = 'inputCols'
  static OUTPUT_COLS_UNIFORM_NAME = 'outputCols'
  static INPUT_COL_PAD_UNIFORM_NAME = 'inputColPad'
  static OUTPUT_COL_PAD_UNIFORM_NAME = 'outputColPad'
  static RELU_ACTIVATION_UNIFORM_NAME = 'relu'
  static IMAP_ROW_TEXTURE_NAME = 'indexMappingRow'
  static IMAP_COL_TEXTURE_NAME = 'indexMappingCol'

  /**
   * Bind WebGL input textures for input transform operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} indexMappingRow
   * @param {weblas.pipeline.Tensor} indexMappingCol
   */
  _bindInputTexturesInputTransform(input, indexMappingRow, indexMappingCol) {
    const gl = this.webgl.context
    this.numTextures = 3
    this._bindInputTexture(this.inputTransformProgram, input.texture, gl.TEXTURE0, WebGLConv2D.INPUT_TEXTURE_NAME)
    this._bindInputTexture(
      this.inputTransformProgram,
      indexMappingRow.texture,
      gl.TEXTURE1,
      WebGLConv2D.IMAP_ROW_TEXTURE_NAME
    )
    this._bindInputTexture(
      this.inputTransformProgram,
      indexMappingCol.texture,
      gl.TEXTURE2,
      WebGLConv2D.IMAP_COL_TEXTURE_NAME
    )
  }

  /**
   * Bind WebGL input textures for main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} weights
   * @param {weblas.pipeline.Tensor} bias
   */
  _bindInputTexturesMain(input, weights, bias) {
    const gl = this.webgl.context
    this.numTextures = 3
    this._bindInputTexture(this.mainProgram, input.texture, gl.TEXTURE0, WebGLConv2D.INPUT_TEXTURE_NAME)
    this._bindInputTexture(this.mainProgram, weights.texture, gl.TEXTURE1, WebGLConv2D.WEIGHTS_TEXTURE_NAME)
    this._bindInputTexture(this.mainProgram, bias.texture, gl.TEXTURE2, WebGLConv2D.BIAS_TEXTURE_NAME)
  }

  /**
   * Bind WebGL uniforms for input transform operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} indexMappingRow
   */
  _bindUniformsInputTransform(input, indexMappingRow) {
    const gl = this.webgl.context
    const nbPatches = input.shape[0]
    const patchLen = input.shape[1]
    const nbFilter = indexMappingRow.shape[1]
    const inputColPad = this.webgl.getPad(patchLen)
    const outputColPad = this.webgl.getPad(nbFilter)
    gl.uniform1i(gl.getUniformLocation(this.inputTransformProgram, WebGLConv2D.INPUT_ROWS_UNIFORM_NAME), nbPatches)
    gl.uniform1i(gl.getUniformLocation(this.inputTransformProgram, WebGLConv2D.INPUT_COLS_UNIFORM_NAME), patchLen)
    gl.uniform1i(gl.getUniformLocation(this.inputTransformProgram, WebGLConv2D.OUTPUT_COLS_UNIFORM_NAME), nbFilter)
    gl.uniform1i(gl.getUniformLocation(this.inputTransformProgram, WebGLConv2D.INPUT_COL_PAD_UNIFORM_NAME), inputColPad)
    gl.uniform1i(
      gl.getUniformLocation(this.inputTransformProgram, WebGLConv2D.OUTPUT_COL_PAD_UNIFORM_NAME),
      outputColPad
    )
  }

  /**
   * Bind WebGL uniforms for main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} weights
   * @param {string} activation
   */
  _bindUniformsMain(input, weights, activation) {
    const gl = this.webgl.context
    const nbFilter = weights.shape[0]
    const patchLen = input.shape[1]
    const inputColPad = this.webgl.getPad(patchLen)
    const outputColPad = this.webgl.getPad(nbFilter)
    gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.INPUT_COLS_UNIFORM_NAME), patchLen)
    gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.OUTPUT_COLS_UNIFORM_NAME), nbFilter)
    gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.INPUT_COL_PAD_UNIFORM_NAME), inputColPad)
    gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.OUTPUT_COL_PAD_UNIFORM_NAME), outputColPad)
    if (activation === 'relu') {
      gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.RELU_ACTIVATION_UNIFORM_NAME), 1)
    }
  }

  /**
   * Bind WebGL output texture for input transform operation.
   *
   * @param {weblas.pipeline.Tensor} indexMappingRow
   * @param {weblas.pipeline.Tensor} inputTransformed
   */
  _bindOutputTextureInputTransform(indexMappingRow, inputTransformed) {
    const nbFilter = indexMappingRow.shape[1]
    const outputColPad = this.webgl.getPad(nbFilter)
    this.webgl.bindOutputTexture(indexMappingRow.shape[0], (nbFilter + outputColPad) / 4, inputTransformed.texture)
  }

  /**
   * Bind WebGL output texture for main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} weights
   * @param {weblas.pipeline.Tensor} tOut
   */
  _bindOutputTextureMain(input, weights, tOut) {
    const nbPatches = input.shape[0]
    const nbFilter = weights.shape[0]
    const outputColPad = this.webgl.getPad(nbFilter)
    this.webgl.bindOutputTexture(nbPatches, (nbFilter + outputColPad) / 4, tOut.texture)
  }

  /**
   * Transform input operation.
   * indexMappingRow and indexMappingCol contain index mappings on the encoded input
   * matrix.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} indexMappingRow
   * @param {weblas.pipeline.Tensor} indexMappingCol
   *
   * @returns {weblas.pipeline.Tensor}
   */
  transformInput(input, indexMappingRow, indexMappingCol) {
    if (
      indexMappingRow.shape[0] !== indexMappingCol.shape[0] ||
      indexMappingRow.shape[1] !== indexMappingCol.shape[1]
    ) {
      throw new Error('Invalid indexMappingRow or indexMappingCol weblas tensor shapes.')
    }

    this.webgl.selectProgram(this.inputTransformProgram)

    const inputTransformed = new weblas.pipeline.Tensor(indexMappingRow.shape, null)

    this._bindInputTexturesInputTransform(input, indexMappingRow, indexMappingCol)
    this._bindUniformsInputTransform(input, indexMappingRow)
    this._bindOutputTextureInputTransform(indexMappingRow, inputTransformed)
    this._compute()
    this._unbindInputTextures()

    return inputTransformed
  }

  /**
   * Main operation.
   *
   * @param {weblas.pipeline.Tensor} input
   * @param {weblas.pipeline.Tensor} weights
   * @param {weblas.pipeline.Tensor} bias
   * @param {string} activation
   * @param {weblas.pipeline.Tensor} [indexMappingRow]
   * @param {weblas.pipeline.Tensor} [indexMappingCol]
   *
   * @returns {weblas.pipeline.Tensor}
   */
  call(input, weights, bias, activation, indexMappingRow, indexMappingCol) {
    if (indexMappingRow && indexMappingCol) {
      input = this.transformInput(input, indexMappingRow, indexMappingCol)
    }
    if (input.shape[1] !== weights.shape[1]) {
      throw new Error('Invalid input or weights weblas tensor shapes.')
    }

    this.webgl.selectProgram(this.mainProgram)

    const nbPatches = input.shape[0]
    const nbFilter = weights.shape[0]

    // create an empty output Tensor
    const tOut = new weblas.pipeline.Tensor([nbPatches, nbFilter], null)

    this._bindInputTexturesMain(input, weights, bias)
    this._bindUniformsMain(input, weights, activation)
    this._bindOutputTextureMain(input, weights, tOut)
    this._compute()
    this._unbindInputTextures()

    return tOut
  }
}

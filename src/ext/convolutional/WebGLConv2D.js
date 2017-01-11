import WebGLLayer from '../WebGLLayer'

export default class WebGLConv2D extends WebGLLayer {
  constructor () {
    super()
    this.inputTransformProgram = this.webgl.createProgram(
      require('./input_transform.glsl')
    )
    this.mainProgram = this.webgl.createProgram(
      require('./conv2d.glsl')
    )
  }

  static INPUT_TEXTURE_UNIFORM_NAME = 'A'
  static WEIGHTS_TEXTURE_UNIFORM_NAME = 'B_t'
  static BIAS_TEXTURE_UNIFORM_NAME = 'C'
  static SHARED_LENGTH_UNIFORM_NAME = 'K'
  static COLUMN_COUNT_UNIFORM_NAME = 'N'
  static PAD_UNIFORM_NAME = 'pad'
  static RELU_ACTIVATION_UNIFORM_NAME = 'relu'
  static IMAP_ROW_TEXTURE_UNIFORM_NAME = 'indexMappingRow'
  static IMAP_COL_TEXTURE_UNIFORM_NAME = 'indexMappingCol'

  transformInput (input, indexMappingRow, indexMappingCol) {
    if (indexMappingRow.shape[0] !== indexMappingCol.shape[0] ||
      indexMappingRow.shape[1] !== indexMappingCol.shape[1]
    ) {
      throw new Error('Invalid indexMappingRow or indexMappingCol weblas tensor shapes.')
    }

    this.webgl.selectProgram(this.inputTransformProgram)
    const gl = this.webgl.context

    const inputTransformed = new weblas.pipeline.Tensor(indexMappingRow.shape, null)

    this._bindInputTexture(this.inputTransformProgram, input.texture, gl.TEXTURE0, WebGLConv2D.INPUT_TEXTURE_UNIFORM_NAME)
    this._bindInputTexture(this.inputTransformProgram, indexMappingRow.texture, gl.TEXTURE1, WebGLConv2D.IMAP_ROW_TEXTURE_UNIFORM_NAME)
    this._bindInputTexture(this.inputTransformProgram, indexMappingCol.texture, gl.TEXTURE2, WebGLConv2D.IMAP_COL_TEXTURE_UNIFORM_NAME)

    const pad = this.webgl.getPad(indexMappingRow.shape[1])

    gl.uniform1i(gl.getUniformLocation(this.inputTransformProgram, WebGLConv2D.COLUMN_COUNT_UNIFORM_NAME), indexMappingRow.shape[1])
    gl.uniform1i(gl.getUniformLocation(this.inputTransformProgram, WebGLConv2D.PAD_UNIFORM_NAME), pad)

    this.webgl.bindOutputTexture(indexMappingRow.shape[0], (indexMappingRow.shape[1] + pad) / 4, inputTransformed.texture)

    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)

    this.webgl.unbindInputTexture(gl.TEXTURE0)
    this.webgl.unbindInputTexture(gl.TEXTURE1)
    this.webgl.unbindInputTexture(gl.TEXTURE2)

    return inputTransformed
  }

  call (input, weights, bias, activation, indexMappingRow, indexMappingCol) {
    console.log('      --', input.shape, weights.shape)
    if (indexMappingRow && indexMappingCol) {
      input = this.transformInput(input, indexMappingRow, indexMappingCol)
    }
    console.log('      ++', input.shape, weights.shape)
    if (input.shape[1] !== weights.shape[1]) {
      throw new Error('Invalid input or weights weblas tensor shapes.')
    }

    this.webgl.selectProgram(this.mainProgram)
    const gl = this.webgl.context

    const M = input.shape[0]
    const N = weights.shape[0]
    const K = input.shape[1]

    // create an empty output Tensor
    const tOut = new weblas.pipeline.Tensor([M, N], null)

    // bind our input textures containing matrix data
    this._bindInputTexture(this.mainProgram, input.texture, gl.TEXTURE0, WebGLConv2D.INPUT_TEXTURE_UNIFORM_NAME)
    this._bindInputTexture(this.mainProgram, weights.texture, gl.TEXTURE1, WebGLConv2D.WEIGHTS_TEXTURE_UNIFORM_NAME)
    this._bindInputTexture(this.mainProgram, bias.texture, gl.TEXTURE2, WebGLConv2D.BIAS_TEXTURE_UNIFORM_NAME)

    const kPad = this.webgl.getPad(K)
    const nPad = this.webgl.getPad(N)

    // bind uniforms
    gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.SHARED_LENGTH_UNIFORM_NAME), K + kPad)
    gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.COLUMN_COUNT_UNIFORM_NAME), N)
    gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.PAD_UNIFORM_NAME), nPad)
    if (activation === 'relu') {
      gl.uniform1i(gl.getUniformLocation(this.mainProgram, WebGLConv2D.RELU_ACTIVATION_UNIFORM_NAME), 1)
    }

    // create our destination texture
    this.webgl.bindOutputTexture(M, (N + nPad) / 4, tOut.texture)

    // initiate calculation
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)

    this.webgl.unbindInputTexture(gl.TEXTURE0)
    this.webgl.unbindInputTexture(gl.TEXTURE1)
    this.webgl.unbindInputTexture(gl.TEXTURE2)

    return tOut
  }
}

import WebGLLayer from '../WebGLLayer'
import range from 'lodash/range'
import sum from 'lodash/sum'

const MODE_CODE = { sum: 0, mul: 1, concat: 2, ave: 3, max: 4 }

export default class WebGLMerge extends WebGLLayer {
  constructor(mode) {
    super()
    if (mode === 'concat') {
      this.program = this.webgl.createProgram(require('./merge_concat.glsl'))
    } else if (['sum', 'mul', 'ave', 'max'].indexOf(mode) > -1) {
      this.program = this.webgl.createProgram(require('./merge.glsl'))
    } else {
      throw new Error(`${mode} mode currently not supported in WebGLMerge layer.`)
    }

    this.mode = mode
    this.modeCode = MODE_CODE[mode]
  }

  static INPUT_TEXTURES_ARRAY_NAME = 'inputs'
  static INPUT_CHANNEL_START_INDICES_UNIFORM_NAME = 'inputChannelStartIndices'
  static NUM_INPUTS_UNIFORM_NAME = 'numInputs'
  static MODE_CODE_UNIFORM_NAME = 'modeCode'
  static OUTPUT_ROWS_UNIFORM_NAME = 'outputRows'
  static OUTPUT_COLS_UNIFORM_NAME = 'outputCols'
  static OUTPUT_COL_PAD_UNIFORM_NAME = 'outputColPad'

  /**
   * Bind WebGL input textures array with a single uniform name.
   *
   * @param {weblas.pipeline.Tensor[]} inputs
   */
  _bindInputTexturesArray(inputs) {
    if (inputs.length > this.MAX_NUM_TEXTURES) {
      throw new Error('Max number of inputs to WebGLMerge exceeded.')
    }

    const gl = this.webgl.context
    this.numTextures = inputs.length

    for (let i = 0; i < inputs.length; i++) {
      gl.activeTexture(gl.TEXTURE0 + i)
      gl.bindTexture(gl.TEXTURE_2D, inputs[i].texture)
    }

    const sampler = gl.getUniformLocation(this.program, `${WebGLMerge.INPUT_TEXTURES_ARRAY_NAME}[0]`)
    gl.uniform1iv(sampler, range(this.numTextures))
  }

  /**
   * Bind WebGL uniforms for main operation.
   *
   * @param {weblas.pipeline.Tensor[]} inputs
   */
  _bindUniforms(inputs) {
    const gl = this.webgl.context

    const outputCols = inputs[0].shape[1]
    const outputColPad = this.webgl.getPad(outputCols)

    gl.uniform1i(gl.getUniformLocation(this.program, WebGLMerge.NUM_INPUTS_UNIFORM_NAME), inputs.length)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLMerge.OUTPUT_COLS_UNIFORM_NAME), outputCols)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLMerge.OUTPUT_COL_PAD_UNIFORM_NAME), outputColPad)

    if (this.mode === 'concat') {
      // with concat, inputs are first transposed to be channel-first
      const inputChannelStartIndices = inputs
        .map(x => x.shape[0])
        .reduce(
          (arr, dim) => {
            if (arr.length > 1) {
              dim += arr[arr.length - 1]
            }
            arr.push(dim)
            return arr
          },
          [0]
        )
        .slice(0, -1)

      const outputRows = sum(inputs.map(x => x.shape[0]))
      gl.uniform1i(gl.getUniformLocation(this.program, WebGLMerge.OUTPUT_ROWS_UNIFORM_NAME), outputRows)
      gl.uniform1iv(
        gl.getUniformLocation(this.program, WebGLMerge.INPUT_CHANNEL_START_INDICES_UNIFORM_NAME),
        inputChannelStartIndices
      )
    } else {
      gl.uniform1i(gl.getUniformLocation(this.program, WebGLMerge.MODE_CODE_UNIFORM_NAME), this.modeCode)
    }
  }

  /**
   * Bind WebGL output texture for main operation.
   *
   * @param {weblas.pipeline.Tensor[]} inputs
   * @param {weblas.pipeline.Tensor} tOut
   */
  _bindOutputTexture(inputs, tOut) {
    let outputRows = inputs[0].shape[0]
    if (this.mode === 'concat') {
      outputRows = sum(inputs.map(x => x.shape[0]))
    }
    const outputCols = inputs[0].shape[1]
    const outputColPad = this.webgl.getPad(outputCols)
    this.webgl.bindOutputTexture(outputRows, (outputCols + outputColPad) / 4, tOut.texture)
  }

  /**
   * Main operation.
   *
   * @param {weblas.pipeline.Tensor[]} inputs
   * @returns {weblas.pipeline.Tensor}
   */
  call(inputs) {
    this.webgl.selectProgram(this.program)

    // create an empty output Tensor
    let tOut
    if (this.mode === 'concat') {
      // concat along channel axis
      if (!inputs.every(x => x.shape[0] === inputs[0].shape[0])) {
        throw new Error('Non-concat axis dimension of inputs to WebGLMerge must all be the same.')
      }
      // for fragment shader ease-of-operation, we first transpose weblas tensors
      // into shape with channels as rows
      const inputsTransposed = inputs.map(x => x.transpose())
      const newShape = [sum(inputsTransposed.map(x => x.shape[0])), inputsTransposed[0].shape[1]]
      tOut = new weblas.pipeline.Tensor(newShape, null)

      // must select WebGL program again since differen program was loaded during transpose
      this.webgl.selectProgram(this.program)

      this._bindInputTexturesArray(inputsTransposed)
      this._bindUniforms(inputsTransposed)
      this._bindOutputTexture(inputsTransposed, tOut)
    } else {
      tOut = new weblas.pipeline.Tensor(inputs[0].shape, null)

      this._bindInputTexturesArray(inputs)
      this._bindUniforms(inputs)
      this._bindOutputTexture(inputs, tOut)
    }

    this._compute()
    this._unbindInputTextures()

    if (this.mode === 'concat') {
      // re-transpose weblas tensor into shape with channels as columns
      tOut = tOut.transpose()
    }

    return tOut
  }
}

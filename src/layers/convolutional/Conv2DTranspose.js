import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../activations'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'

/**
 * Conv2DTranspose layer class
 */
export default class Conv2DTranspose extends Layer {
  /**
   * Creates a Conv2DTranspose layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.filters] - Number of convolution filters to use.
   * @param {number|number[]} [attrs.kernel_size] - Size of the convolution kernel.
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Conv2DTranspose'

    const {
      filters = 1,
      kernel_size = [3, 3],
      strides = [1, 1],
      padding = 'valid',
      data_format = 'channels_last',
      activation = 'linear',
      use_bias = true
    } = attrs

    if (Array.isArray(kernel_size)) {
      this.kernelShape = [filters, ...kernel_size]
    } else {
      this.kernelShape = [filters, kernel_size, kernel_size]
    }

    if (Array.isArray(strides)) {
      this.strides = strides
    } else {
      this.strides = [strides, strides]
    }

    if (padding === 'valid' || padding === 'same') {
      this.padding = padding
    } else {
      throw new Error(`${this.name} [Conv2DTranspose layer] Invalid padding.`)
    }

    if (data_format === 'channels_last' || data_format === 'channels_first') {
      this.dataFormat = data_format
    } else {
      throw new Error(
        `${this.name} [Conv2DTranspose layer] Only channels_last and channels_first data formats are allowed.`
      )
    }

    this.activation = activation
    this.activationFunc = activations[activation]

    this.use_bias = use_bias

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel']

    // GPU setup
    if (this.gpu) {
      this.matMulProgram = webgl2.compileProgram(require('../../matMul.webgl2.glsl'))
      this.convTransposeProgram = webgl2.compileProgram(require('./Conv2DTranspose.webgl2.glsl'))
      this.activationProgram = webgl2.compileProgram(require(`../../activations/${this.activation}.webgl2.glsl`))
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   *
   * W weight tensor is converted to `channels_last` mode if in `channels_first` mode.
   *
   * In `channels_last` mode, W weight tensor has shape [nbRow, nbCol, inputChannels, nbFilter]
   *
   * In `channels_first` mode, W weight tensor has shape [nbFilter, inputChannels, nbRow, nbCol]
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    if (this.dataFormat === 'channels_first') {
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(2, 3, 1, 0)
    }
    super.setWeights(weightsArr, false)

    this._w2row()

    if (this.gpu) {
      this.weights['kernel'] = this._wRowsMat
      this.weights['kernel'].createGLTexture()
      if (this.use_bias) {
        this.weights['bias'].createGLTexture()
      }
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
   * Method for computing output dimensions and padding, based on input dimensions, kernel size, and padding mode.
   *
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   *
   * For deconvolution, we will "take away" padding from the output rather than add padding to the input.
   *
   * For more details on calculating output shapes and padding for transposed convolutions (deconvolution here), see:
   * https://arxiv.org/pdf/1603.07285v1.pdf
   *
   * @param {number[]} inputShape
   */
  _calcOutputShape(inputShape) {
    const inputRows = inputShape[0]
    const inputCols = inputShape[1]
    const [nbFilter, nbRow, nbCol] = this.kernelShape

    const outputRows =
      this.padding === 'same'
        ? inputRows * this.strides[0]
        : inputRows * this.strides[0] + Math.max(nbRow - this.strides[0], 0)
    const outputCols =
      this.padding === 'same'
        ? inputCols * this.strides[1]
        : inputCols * this.strides[1] + Math.max(nbCol - this.strides[1], 0)
    const outputChannels = nbFilter

    const paddingRow =
      this.padding === 'same' ? Math.max(0, Math.floor((inputRows - 1) * this.strides[0] + nbRow - outputRows)) : 0
    const paddingCol =
      this.padding === 'same' ? Math.max(0, Math.floor((inputCols - 1) * this.strides[1] + nbCol - outputCols)) : 0
    const paddingRowBefore = Math.floor(paddingRow / 2)
    const paddingRowAfter = paddingRow - paddingRowBefore
    const paddingColBefore = Math.floor(paddingCol / 2)
    const paddingColAfter = paddingCol - paddingColBefore

    this.outputShape = [outputRows, outputCols, outputChannels]
    this.outputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter]
  }

  /**
   * Convert input image to column matrix, along channels axis
   *
   * shape: [inputRows, inputCols, inputChannels] -> [inputRows * inputCols, inputChannels]
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  _im2col(x) {
    const [inputRows, inputCols, inputChannels] = x.tensor.shape

    if (!this._imColsMat) {
      this._imColsMat = new Tensor([], [inputRows * inputCols, inputChannels])
    }

    let channelRaveled = new Tensor([], [inputRows * inputCols])
    let channel = new Tensor([], [inputRows, inputCols])
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(channel.tensor, x.tensor.pick(null, null, c))
      channelRaveled.replaceTensorData(channel.tensor.data)
      ops.assign(this._imColsMat.tensor.pick(null, c), channelRaveled.tensor)
    }
    if (this.gpu) {
      this._imColsMat.createGLTexture()
    }
    return this._imColsMat
  }

  /**
   * Convert filter weights to row matrix, along channels axis
   *
   * shape: [nbRow, nbCol, nbFilter, inputChannels] -> [inputChannels, nbRow * nbCol * nbFilter]
   *
   * @returns {Tensor}
   */
  _w2row() {
    const [nbRow, nbCol, nbFilter, inputChannels] = this.weights['kernel'].tensor.shape

    this._wRowsMat = new Tensor([], [inputChannels, nbRow * nbCol * nbFilter])

    let channelRaveled = new Tensor([], [nbRow * nbCol * nbFilter])
    let channel = new Tensor([], [nbRow, nbCol, nbFilter])
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(channel.tensor, this.weights['kernel'].tensor.pick(null, null, null, c))
      channelRaveled.replaceTensorData(channel.tensor.data)
      ops.assign(this._wRowsMat.tensor.pick(c, null), channelRaveled.tensor)
    }

    return this._wRowsMat
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _call_cpu(x) {
    this.inputShape = x.tensor.shape
    this._calcOutputShape(this.inputShape)
    this._im2col(x)

    const inputRows = x.tensor.shape[0]
    const inputCols = x.tensor.shape[1]
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    const matMul = new Tensor([], [inputRows * inputCols, nbRow * nbCol * nbFilter])

    gemm(matMul.tensor, this._imColsMat.tensor, this._wRowsMat.tensor, 1, 1)

    // add padding which we will take away later
    const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.outputPadding
    this.output = new Tensor([], this.outputShape)
    let outputPadded = new Tensor(
      [],
      [
        this.outputShape[0] + paddingRowBefore + paddingRowAfter,
        this.outputShape[1] + paddingColBefore + paddingColAfter,
        this.outputShape[2]
      ]
    )

    const patchShape = [nbRow, nbCol, nbFilter]
    let patch = new Tensor([], patchShape)
    let patchRaveled = new Tensor([], [nbRow * nbCol * nbFilter])
    let index = 0
    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        ops.assign(patchRaveled.tensor, matMul.tensor.pick(index, null))
        patch.replaceTensorData(patchRaveled.tensor.data)
        const iOutPos = i * this.strides[0]
        const jOutPos = j * this.strides[1]
        ops.addeq(
          outputPadded.tensor.hi(iOutPos + nbRow, jOutPos + nbCol, this.outputShape[2]).lo(iOutPos, jOutPos, 0),
          patch.tensor
        )
        index += 1
      }
    }

    // remove padding
    ops.assign(
      this.output.tensor,
      outputPadded.tensor
        .hi(this.outputShape[0] + paddingRowBefore, this.outputShape[1] + paddingColBefore, this.outputShape[2])
        .lo(paddingRowBefore, paddingColBefore, 0)
    )

    // bias
    if (this.use_bias) {
      for (let n = 0; n < nbFilter; n++) {
        ops.addseq(this.output.tensor.pick(null, null, n), this.weights['bias'].tensor.get(n))
      }
    }

    this.activationFunc(this.output)

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      this.output.tensor = this.output.tensor.transpose(2, 0, 1)
    }
  }

  /**
   * In GPU mode, we work directly on 2D-tiled representations of the tensors. After the matrix multiply step produce
   * matrix Y, the final output Z at coordinate [i,j] will be the summation of a number of elements of the matrix Y.
   * Here, we calculate the indices of matrix Y for each coordinate [i,j] of Z, and encode these index maps as texture
   * arrays.
   *
   * @param {number[]} inputShape
   */
  _createIndexMaps(inputShape) {
    if (this._tiledOutputRowIndicesMap && this._tiledOutputColIndicesMap) {
      return
    }

    const inputRows = inputShape[0]
    const inputCols = inputShape[1]
    const [nbFilter, nbRow, nbCol] = this.kernelShape

    const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.outputPadding

    const indicesMapShape = [...this.outputShape, inputRows * inputCols]
    const indicesMapShapePadded = [
      this.outputShape[0] + paddingRowBefore + paddingRowAfter,
      this.outputShape[1] + paddingColBefore + paddingColAfter,
      this.outputShape[2],
      inputRows * inputCols
    ]
    const outputRowIndicesMap = new Tensor([], indicesMapShape, { type: Int32Array })
    const outputColIndicesMap = new Tensor([], indicesMapShape, { type: Int32Array })
    const outputRowIndicesMapPadded = new Tensor([], indicesMapShapePadded, { type: Int32Array })
    const outputColIndicesMapPadded = new Tensor([], indicesMapShapePadded, { type: Int32Array })
    ops.assigns(outputRowIndicesMap.tensor, -1)
    ops.assigns(outputColIndicesMap.tensor, -1)
    ops.assigns(outputRowIndicesMapPadded.tensor, -1)
    ops.assigns(outputColIndicesMapPadded.tensor, -1)

    const matMulColIndicesPatch = new Tensor([], [nbRow, nbCol, nbFilter, 1], { type: Int32Array })
    for (let i = 0; i < nbRow * nbCol * nbFilter; i++) {
      matMulColIndicesPatch.tensor.data[i] = i
    }

    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        const matMulRowIndex = i * inputCols + j
        const iOutPos = i * this.strides[0]
        const jOutPos = j * this.strides[1]
        ops.assigns(
          outputRowIndicesMapPadded.tensor
            .hi(iOutPos + nbRow, jOutPos + nbCol, this.outputShape[2], matMulRowIndex + 1)
            .lo(iOutPos, jOutPos, 0, matMulRowIndex),
          matMulRowIndex
        )
        ops.assign(
          outputColIndicesMapPadded.tensor
            .hi(iOutPos + nbRow, jOutPos + nbCol, this.outputShape[2], matMulRowIndex + 1)
            .lo(iOutPos, jOutPos, 0, matMulRowIndex),
          matMulColIndicesPatch.tensor
        )
      }
    }

    // remove padding
    ops.assign(
      outputRowIndicesMap.tensor,
      outputRowIndicesMapPadded.tensor
        .hi(
          this.outputShape[0] + paddingRowBefore,
          this.outputShape[1] + paddingColBefore,
          this.outputShape[2],
          inputRows * inputCols
        )
        .lo(paddingRowBefore, paddingColBefore, 0, 0)
    )
    ops.assign(
      outputColIndicesMap.tensor,
      outputColIndicesMapPadded.tensor
        .hi(
          this.outputShape[0] + paddingRowBefore,
          this.outputShape[1] + paddingColBefore,
          this.outputShape[2],
          inputRows * inputCols
        )
        .lo(paddingRowBefore, paddingColBefore, 0, 0)
    )
    // combine first two dimensions
    const tiledIndicesMapShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2], inputRows * inputCols]
    this._tiledOutputRowIndicesMap = new Tensor([], tiledIndicesMapShape, { type: Int32Array })
    this._tiledOutputColIndicesMap = new Tensor([], tiledIndicesMapShape, { type: Int32Array })
    const channelData = new Tensor([], [this.outputShape[2], inputRows * inputCols], { type: Int32Array })
    for (let i = 0; i < this.outputShape[0]; i++) {
      for (let j = 0; j < this.outputShape[1]; j++) {
        ops.assign(channelData.tensor, outputRowIndicesMap.tensor.pick(i, j, null, null))
        ops.assign(
          this._tiledOutputRowIndicesMap.tensor.pick(i * this.outputShape[1] + j, null, null),
          channelData.tensor
        )
        ops.assign(channelData.tensor, outputColIndicesMap.tensor.pick(i, j, null, null))
        ops.assign(
          this._tiledOutputColIndicesMap.tensor.pick(i * this.outputShape[1] + j, null, null),
          channelData.tensor
        )
      }
    }

    if (this.gpu) {
      this._tiledOutputRowIndicesMap.createGLTexture('2d_array', 'int')
      this._tiledOutputColIndicesMap.createGLTexture('2d_array', 'int')
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _call_gpu(x) {
    if (x.glTextureIsTiled) {
      this.inputShape = x.untiledShape
      this._calcOutputShape(this.inputShape)
    } else {
      this.inputShape = x.tensor.shape
      this._calcOutputShape(this.inputShape)
      this._im2col(x)
      x.glTexture = this._imColsMat.glTexture
      x.glTextureShape = this._imColsMat.glTextureShape
    }

    // create output textures if doesn't already exist
    if (!this.output_matmul) {
      const outputTextureShape = [x.glTextureShape[0], this.weights['kernel'].glTextureShape[1]]
      this.output_matmul = new Tensor([], outputTextureShape)
      this.output_matmul.createGLTexture()
    }
    if (!this.outputPreactiv) {
      const outputTextureShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]]
      this.outputPreactiv = new Tensor([], outputTextureShape)
      this.outputPreactiv.createGLTexture()
      this.outputPreactiv.glTextureIsTiled = true
      this.outputPreactiv.untiledShape = this.outputShape
    }
    if (!this.output) {
      const outputTextureShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]]
      this.output = new Tensor([], outputTextureShape)
      this.output.createGLTexture()
      this.output.glTextureIsTiled = true
      this.output.untiledShape = this.outputShape
    }

    // Matrix Multiply with kernel
    webgl2.selectProgram(this.matMulProgram)
    webgl2.bindOutputTexture(this.output_matmul.glTexture, this.output_matmul.glTextureShape)
    let textures = [x.glTexture, this.weights['kernel'].glTexture]
    let textureTypes = ['2d', '2d']
    let textureNames = ['A', 'B']
    webgl2.bindInputTextures(this.matMulProgram, textures, textureTypes, textureNames)
    let uniforms = [0, x.glTextureShape[0], ...this.weights['kernel'].glTextureShape]
    let uniformTypes = ['bool', 'int', 'int', 'int']
    let uniformNames = ['addC', 'M', 'K', 'N']
    webgl2.bindUniforms(this.matMulProgram, uniforms, uniformTypes, uniformNames)
    webgl2.runProgram()

    // Tranposed Convolution
    this._createIndexMaps(this.inputShape)
    const test = new Tensor([], [this.outputShape[0] * this.outputShape[1], this.outputShape[2]])
    ops.assign(test.tensor, this._tiledOutputRowIndicesMap.tensor.pick(null, null, 0))
    webgl2.selectProgram(this.convTransposeProgram)
    webgl2.bindOutputTexture(this.outputPreactiv.glTexture, this.outputPreactiv.glTextureShape)
    textures = [
      this.output_matmul.glTexture,
      this._tiledOutputRowIndicesMap.glTexture,
      this._tiledOutputColIndicesMap.glTexture
    ]
    textureTypes = ['2d', '2d_array', '2d_array']
    textureNames = ['matMulOutput', 'rowIndicesMap', 'colIndicesMap']
    if (this.use_bias) {
      textures.push(this.weights['bias'].glTexture)
      textureTypes.push('2d')
      textureNames.push('bias')
    }
    webgl2.bindInputTextures(this.convTransposeProgram, textures, textureTypes, textureNames)
    uniforms = [
      this.use_bias ? 1 : 0,
      this.outputShape[0] * this.outputShape[1],
      this.outputShape[2],
      this.inputShape[0] * this.inputShape[1]
    ]
    uniformTypes = ['bool', 'int', 'int', 'int']
    uniformNames = ['use_bias', 'rows', 'cols', 'summationLength']
    webgl2.bindUniforms(this.convTransposeProgram, uniforms, uniformTypes, uniformNames)
    webgl2.runProgram()

    // Activation
    if (this.activation === 'linear') {
      this.output = this.outputPreactiv
    } else {
      webgl2.selectProgram(this.activationProgram)
      webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
      textures = [this.outputPreactiv.glTexture]
      textureTypes = ['2d']
      textureNames = ['x']
      webgl2.bindInputTextures(this.activationProgram, textures, textureTypes, textureNames)
      webgl2.runProgram()
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
      this.output.reshapeTensorFromTiled()

      // convert back to channels_first ordering if necessary
      if (this.dataFormat === 'channels_first') {
        this.output.tensor = this.output.tensor.transpose(2, 0, 1)
      }
    }
  }
}

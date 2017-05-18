import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'

/**
 * Conv2DTranspose layer class
 */
export default class Conv2DTranspose extends Layer {
  /**
   * Creates a Conv2DTranspose layer
   * @param {Number} attrs.filters - Number of convolution filters to use.
   * @param {Array<Number>|Number} attrs.kernel_size - Size of the convolution kernel.
   * @param {Object} [attrs] - layer attributes
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

    // Enable layer gpu +/- pipeline mode if supported
    if (this.gpu && weblas) {
      this._useWeblas = true
      this._pipelineEnabled = false
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   * W weight tensor is converted to `channels_last` mode if in `channels_first` mode.
   * In `channels_last` mode, W weight tensor has shape [nbRow, nbCol, inputChannels, nbFilter]
   * In `channels_first` mode, W weight tensor has shape [nbFilter, inputChannels, nbRow, nbCol]
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    if (this.dataFormat === 'channels_first') {
      // W
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(2, 3, 1, 0)
    }
    super.setWeights(weightsArr)

    this._wRowsMat = this._w2row()
    if (this._useWeblas) {
      this._wRowsMat.createWeblasTensor()
      if (!this._wRowsMat._gpuMaxSizeExceeded) {
        this._wRowsMat.weblasTensor = this._wRowsMat.weblasTensor.transpose()
      }
    }
  }

  /**
   * Method for computing output dimensions and padding, based on input
   * dimensions, kernel size, and padding mode.
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   * For deconvolution, we will "take away" padding from the output rather than add padding
   * to the input.
   * For more details on calculating output shapes and padding for transposed convolutions
   * (deconvolution here), see: https://arxiv.org/pdf/1603.07285v1.pdf
   * @param {number[]} inputShape
   */
  _calcOutputShape(inputShape) {
    const inputRows = inputShape[0]
    const inputCols = inputShape[1]
    const [nbFilter, nbRow, nbCol] = this.kernelShape

    const outputRows = this.padding === 'same'
      ? inputRows * this.strides[0]
      : inputRows * this.strides[0] + Math.max(nbRow - this.strides[0], 0)
    const outputCols = this.padding === 'same'
      ? inputCols * this.strides[1]
      : inputCols * this.strides[1] + Math.max(nbCol - this.strides[1], 0)
    const outputChannels = nbFilter

    const paddingRow = this.padding === 'same'
      ? Math.max(0, Math.floor((inputRows - 1) * this.strides[0] + nbRow - outputRows))
      : 0
    const paddingCol = this.padding === 'same'
      ? Math.max(0, Math.floor((inputCols - 1) * this.strides[1] + nbCol - outputCols))
      : 0
    const paddingRowBefore = Math.floor(paddingRow / 2)
    const paddingRowAfter = paddingRow - paddingRowBefore
    const paddingColBefore = Math.floor(paddingCol / 2)
    const paddingColAfter = paddingCol - paddingColBefore

    this.outputShape = [outputRows, outputCols, outputChannels]
    this.outputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter]
  }

  /**
   * Convert input image to column matrix, along channels axis
   * shape: [inputRows, inputCols, inputChannels] -> [inputRows * inputCols, inputChannels]
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _im2col(x) {
    const [inputRows, inputCols, inputChannels] = x.tensor.shape

    const imColsMat = new Tensor([], [inputRows * inputCols, inputChannels])
    let channelRaveled = new Tensor([], [inputRows * inputCols])
    let channel = new Tensor([], [inputRows, inputCols])
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(channel.tensor, x.tensor.pick(null, null, c))
      channelRaveled.replaceTensorData(channel.tensor.data)
      ops.assign(imColsMat.tensor.pick(null, c), channelRaveled.tensor)
    }
    return imColsMat
  }

  /**
   * Convert filter weights to row matrix, along channels axis
   * shape: [nbRow, nbCol, nbFilter, inputChannels] -> [inputChannels, nbRow * nbCol * nbFilter]
   * @returns {Tensor|weblas.pipeline.Tensor} wRowsMat
   */
  _w2row() {
    const [nbRow, nbCol, nbFilter, inputChannels] = this.weights['kernel'].tensor.shape

    const wRowsMat = new Tensor([], [inputChannels, nbRow * nbCol * nbFilter])

    let channelRaveled = new Tensor([], [nbRow * nbCol * nbFilter])
    let channel = new Tensor([], [nbRow, nbCol, nbFilter])
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(channel.tensor, this.weights['kernel'].tensor.pick(null, null, null, c))
      channelRaveled.replaceTensorData(channel.tensor.data)
      ops.assign(wRowsMat.tensor.pick(c, null), channelRaveled.tensor)
    }

    return wRowsMat
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

    const imColsMat = this._im2col(x)
    if (this._useWeblas) {
      imColsMat.createWeblasTensor()
    }

    const inputRows = x.tensor.shape[0]
    const inputCols = x.tensor.shape[1]
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    const matMul = new Tensor([], [inputRows * inputCols, nbRow * nbCol * nbFilter])
    if (this._useWeblas && !(imColsMat._gpuMaxSizeExceeded || this._wRowsMat._gpuMaxSizeExceeded)) {
      let _zerosVec = new Tensor([], [this.weights['kernel'].tensor.shape[3]])
      _zerosVec.createWeblasTensor()
      matMul.tensor.data = weblas.pipeline
        .sgemm(1, imColsMat.weblasTensor, this._wRowsMat.weblasTensor, 0, _zerosVec)
        .transfer()
      imColsMat.weblasTensor.delete()
      delete imColsMat.weblasTensor
    } else {
      gemm(matMul.tensor, imColsMat.tensor, this._wRowsMat.tensor, 1, 1)
    }

    this._calcOutputShape(x.tensor.shape)

    // add padding which we will take away later
    const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.outputPadding
    let output = new Tensor([], this.outputShape)
    let outputPadded = new Tensor(
      [],
      [
        this.outputShape[0] + paddingRowBefore + paddingRowAfter,
        this.outputShape[1] + paddingColBefore + paddingColAfter,
        this.outputShape[2]
      ]
    )

    // bias
    if (this.use_bias) {
      for (let n = 0; n < nbFilter; n++) {
        ops.assigns(outputPadded.tensor.pick(null, null, n), this.weights['bias'].tensor.get(n))
      }
    }

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
      output.tensor,
      outputPadded.tensor
        .hi(this.outputShape[0] + paddingRowBefore, this.outputShape[1] + paddingColBefore, this.outputShape[2])
        .lo(paddingRowBefore, paddingColBefore, 0)
    )

    x.tensor = output.tensor
    this.activationFunc(x)

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(2, 0, 1)
    }

    return x
  }
}

import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../engine/Layer'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'
import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'

/**
 * AtrousConvolution2D layer class
 */
export default class AtrousConvolution2D extends Layer {
  /**
   * Creates a AtrousConvolution2D layer
   * @param {number} nbFilter - Number of convolution filters to use.
   * @param {number} nbRow - Number of rows in the convolution kernel.
   * @param {number} nbCol - Number of columns in the convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor (nbFilter, nbRow, nbCol, attrs = {}) {
    super(attrs)
    const {
      activation = 'linear',
      borderMode = 'valid',
      subsample = [1, 1],
      atrousRate = [1, 1],
      dimOrdering = 'tf',
      bias = true
    } = attrs

    this.kernelShape = [nbFilter, nbRow, nbCol]

    this.activation = activations[activation]

    if (borderMode === 'valid' || borderMode === 'same') {
      this.borderMode = borderMode
    } else {
      throw new Error(`${this.name} [AtrousConvolution2D layer] Invalid borderMode.`)
    }

    this.subsample = subsample

    this.atrousRate = atrousRate

    if (dimOrdering === 'tf' || dimOrdering === 'th') {
      this.dimOrdering = dimOrdering
    } else {
      throw new Error(`${this.name} [AtrousConvolution2D layer] Only tf and th dim ordering are allowed.`)
    }

    this.bias = bias

    // Layer weights specification
    this.params = this.bias ? ['W', 'b'] : ['W']
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   * W weight tensor is converted to `tf` mode if in `th` mode.
   * In `tf` mode, W weight tensor has shape [nbRow, nbCol, inputChannels, nbFilter]
   * In `th` mode, W weight tensor has shape [nbFilter, inputChannels, nbRow, nbCol]
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights (weightsArr) {
    if (this.dimOrdering === 'th') {
      // W
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(2, 3, 1, 0)
    }
    super.setWeights(weightsArr)
  }

  /**
   * Method for computing output dimensions and padding, based on input
   * dimensions, kernel size, and padding mode.
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   * @param {Tensor} x
   * @returns {number[]} [outputRows, outputCols, outputChannels]
   */
  _calcOutputShape = x => {
    const inputRows = x.tensor.shape[0]
    const inputCols = x.tensor.shape[1]
    const [nbFilter, nbRow, nbCol] = this.kernelShape

    // effective shape after filter dilation
    const nbRowDilated = nbRow + (nbRow - 1) * (this.atrousRate[0] - 1)
    const nbColDilated = nbCol + (nbCol - 1) * (this.atrousRate[1] - 1)

    const outputRows = this.borderMode === 'same'
      ? Math.floor((inputRows + this.subsample[0] - 1) / this.subsample[0])
      : Math.floor((inputRows - nbRowDilated + this.subsample[0]) / this.subsample[0])
    const outputCols = this.borderMode === 'same'
      ? Math.floor((inputCols + this.subsample[1] - 1) / this.subsample[1])
      : Math.floor((inputCols - nbColDilated + this.subsample[1]) / this.subsample[1])
    const outputChannels = nbFilter

    const paddingRow = this.borderMode === 'same'
      ? Math.max(0, Math.floor((outputRows - 1) * this.subsample[0] + nbRowDilated - inputRows))
      : 0
    const paddingCol = this.borderMode === 'same'
      ? Math.max(0, Math.floor((outputCols - 1) * this.subsample[1] + nbColDilated - inputCols))
      : 0
    const paddingRowBefore = Math.floor(paddingRow / 2)
    const paddingRowAfter = paddingRow - paddingRowBefore
    const paddingColBefore = Math.floor(paddingCol / 2)
    const paddingColAfter = paddingCol - paddingColBefore

    this.outputShape = [outputRows, outputCols, outputChannels]
    this.inputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter]
  }

  /**
   * Pad input tensor if necessary, for borderMode='same'.
   * See above for notes on calculating padding.
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _padInput = x => {
    if (this.borderMode === 'same') {
      const [inputRows, inputCols, inputChannels] = x.tensor.shape
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding
      const newRows = inputRows + paddingRowBefore + paddingRowAfter
      const newCols = inputCols + paddingColBefore + paddingColAfter
      let _x = new Tensor([], [newRows, newCols, inputChannels])
      ops.assign(
        _x.tensor
          .hi(inputRows + paddingRowBefore, inputCols + paddingColBefore, inputChannels)
          .lo(paddingRowBefore, paddingColBefore, 0),
        x.tensor
      )
      x.tensor = _x.tensor
    }
    return x
  }

  /**
   * Convert input image to column matrix
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _im2col = x => {
    const [inputRows, inputCols, inputChannels] = x.tensor.shape
    const nbRow = this.kernelShape[1]
    const nbCol = this.kernelShape[2]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const patchLen = nbRow * nbCol * inputChannels

    // effective shape after filter dilation
    const nbRowDilated = nbRow + (nbRow - 1) * (this.atrousRate[0] - 1)
    const nbColDilated = nbCol + (nbCol - 1) * (this.atrousRate[1] - 1)

    const imColsMat = new Tensor([], [nbPatches, patchLen])

    let patch = new Tensor([], [patchLen])
    let n = 0
    for (let i = 0; i <= inputRows - nbRowDilated; i += this.subsample[0]) {
      for (let j = 0; j <= inputCols - nbColDilated; j += this.subsample[1]) {
        const patchData = flattenDeep(unpack(
          x.tensor
            .hi(i + nbRowDilated, j + nbColDilated, inputChannels)
            .lo(i, j, 0)
            .step(this.atrousRate[0], this.atrousRate[1], 1)
        ))
        patch.replaceTensorData(patchData)
        ops.assign(imColsMat.tensor.pick(n, null), patch.tensor)
        n += 1
      }
    }

    return imColsMat
  }

  /**
   * Convert filter weights to row matrix
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _w2row = x => {
    const inputChannels = x.tensor.shape[2]
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    const patchLen = nbRow * nbCol * inputChannels

    const wRowsMat = new Tensor([], [patchLen, nbFilter])

    let patch = new Tensor([], [patchLen])
    for (let n = 0; n < nbFilter; n++) {
      const patchData = flattenDeep(unpack(
        this.weights.W.tensor.pick(null, null, null, n)
      ))
      patch.replaceTensorData(patchData)
      ops.assign(wRowsMat.tensor.pick(null, n), patch.tensor)
    }

    return wRowsMat
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call = x => {
    // convert to tf ordering
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

    this._calcOutputShape(x)
    this._padInput(x)

    const imColsMat = this._im2col(x)
    const wRowsMat = this._w2row(x)

    const nbFilter = this.kernelShape[0]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const matMul = new Tensor([], [nbPatches, nbFilter])
    if (this.bias) {
      for (let n = 0; n < nbFilter; n++) {
        ops.assigns(matMul.tensor.pick(null, n), this.weights.b.tensor.get(n))
      }
    }

    if (x._useWeblas) {
      const bias = this.bias
        ? this.weights.b.tensor.data
        : new Float32Array(wRowsMat.tensor.shape[1])
      matMul.tensor.data = weblas.sgemm(
        imColsMat.tensor.shape[0], wRowsMat.tensor.shape[1], imColsMat.tensor.shape[1], // M, N, K
        1, imColsMat.tensor.data, wRowsMat.tensor.data, // alpha, A, B
        1, bias // beta, C
      )
    } else {
      gemm(matMul.tensor, imColsMat.tensor, wRowsMat.tensor, 1, 1)
    }

    let output = new Tensor([], this.outputShape)
    let outputChannel = new Tensor([], [outputRows, outputCols])
    for (let n = 0; n < nbFilter; n++) {
      const outputChannelData = flattenDeep(unpack(
        matMul.tensor.pick(null, n)
      ))
      outputChannel.replaceTensorData(outputChannelData)
      ops.assign(output.tensor.pick(null, null, n), outputChannel.tensor)
    }
    x.tensor = output.tensor

    this.activation(x)

    // convert back to th ordering if necessary
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(2, 0, 1)
    }

    return x
  }
}

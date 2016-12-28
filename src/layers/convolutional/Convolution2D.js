import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'

/**
 * Convolution2D layer class
 */
export default class Convolution2D extends Layer {
  /**
   * Creates a Convolution2D layer
   * @param {number} attrs.nbFilter - Number of convolution filters to use.
   * @param {number} attrs.nbRow - Number of rows in the convolution kernel.
   * @param {number} attrs.nbCol - Number of columns in the convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor (attrs = {}) {
    super(attrs)
    this.layerClass = 'Convolution2D'

    const {
      nbFilter = 1,
      nbRow = 3,
      nbCol = 3,
      activation = 'linear',
      borderMode = 'valid',
      subsample = [1, 1],
      dimOrdering = 'tf',
      bias = true
    } = attrs

    this.kernelShape = [nbFilter, nbRow, nbCol]

    this.activation = activations[activation]

    if (borderMode === 'valid' || borderMode === 'same') {
      this.borderMode = borderMode
    } else {
      throw new Error(`${this.name} [Convolution2D layer] Invalid borderMode.`)
    }

    this.subsample = subsample

    if (dimOrdering === 'tf' || dimOrdering === 'th') {
      this.dimOrdering = dimOrdering
    } else {
      throw new Error(`${this.name} [Convolution2D layer] Only tf and th dim ordering are allowed.`)
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
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    let shape = weightsArr[0].tensor.shape

    // check for legacy shape of weights
    // Keras:    (nb_filter, input_dim, filter_length, 1)
    // Keras.js: (nbFilter, inputChannels, nbRow, nbCol)
    if (!(shape[0] === nbRow && shape[1] === nbCol) || shape[3] !== nbFilter) {
      console.warn('Using legacy shape of weights')

      if (!(shape[0] === nbFilter & (shape[2] === nbRow & shape[3] === nbCol))) {
        throw new Error('Unsupported shape of weights')
      }
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(2, 3, 1, 0)
    }
    super.setWeights(weightsArr)

    this._w2row()
    if (this._useWeblas) {
      this._wRowsMat.createWeblasTensor()
      if (!this._wRowsMat._gpuMaxSizeExceeded) {
        this._wRowsMat.weblasTensor = this._wRowsMat.weblasTensor.transpose()
      }
      if (this.bias) {
        this.weights.b.createWeblasTensor()
      } else {
        this._zerosVec = new Tensor([], [this.weights.W.tensor.shape[3]])
        this._zerosVec.createWeblasTensor()
      }
    }
  }

  /**
   * Method for computing output dimensions and padding, based on input
   * dimensions, kernel size, and padding mode.
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   * @param {Tensor} x
   */
  _calcOutputShape (x) {
    const inputRows = x.tensor.shape[0]
    const inputCols = x.tensor.shape[1]
    const [nbFilter, nbRow, nbCol] = this.kernelShape

    const outputRows = this.borderMode === 'same'
      ? Math.floor((inputRows + this.subsample[0] - 1) / this.subsample[0])
      : Math.floor((inputRows - nbRow + this.subsample[0]) / this.subsample[0])
    const outputCols = this.borderMode === 'same'
      ? Math.floor((inputCols + this.subsample[1] - 1) / this.subsample[1])
      : Math.floor((inputCols - nbCol + this.subsample[1]) / this.subsample[1])
    const outputChannels = nbFilter

    const paddingRow = this.borderMode === 'same'
      ? Math.max(0, Math.floor((outputRows - 1) * this.subsample[0] + nbRow - inputRows))
      : 0
    const paddingCol = this.borderMode === 'same'
      ? Math.max(0, Math.floor((outputCols - 1) * this.subsample[1] + nbCol - inputCols))
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
  _padInput (x) {
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
  _im2col (x) {
    const [inputRows, inputCols, inputChannels] = x.tensor.shape
    const nbRow = this.kernelShape[1]
    const nbCol = this.kernelShape[2]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const patchLen = nbRow * nbCol * inputChannels

    if (!this._imColsMat) {
      this._imColsMat = new Tensor([], [nbPatches, patchLen])
    }

    if (nbRow === 1 && nbCol === 1 && this.subsample[0] === 1 && this.subsample[1] === 1) {
      this._imColsMat.replaceTensorData(x.tensor.data)
      if (this._useWeblas) {
        this._imColsMat.createWeblasTensor()
      }
      return this._imColsMat
    }

    let patch = new Tensor([], [nbRow, nbCol, inputChannels])
    let offset = 0
    for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.subsample[0]) {
      for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.subsample[1]) {
        ops.assign(patch.tensor, x.tensor.hi(i + nbRow, j + nbCol, inputChannels).lo(i, j, 0))
        this._imColsMat.tensor.data.set(patch.tensor.data, offset)
        offset += patchLen
      }
    }
    if (this._useWeblas) {
      this._imColsMat.createWeblasTensor()
    }
    return this._imColsMat
  }

  /**
   * Convert filter weights to row matrix
   * @returns {Tensor|weblas.pipeline.Tensor} wRowsMat
   */
  _w2row () {
    const inputChannels = this.weights.W.tensor.shape[2]
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    const patchLen = nbRow * nbCol * inputChannels

    this._wRowsMat = new Tensor([], [patchLen, nbFilter])

    let patch = new Tensor([], [nbRow, nbCol, inputChannels])
    let patchRaveled = new Tensor([], [patchLen])
    for (let n = 0; n < nbFilter; n++) {
      ops.assign(patch.tensor, this.weights.W.tensor.pick(null, null, null, n))
      patchRaveled.replaceTensorData(patch.tensor.data)
      ops.assign(this._wRowsMat.tensor.pick(null, n), patchRaveled.tensor)
    }

    return this._wRowsMat
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    // convert to tf ordering
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

    this._calcOutputShape(x)
    this._padInput(x)

    this._im2col(x)

    const nbFilter = this.kernelShape[0]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const matMul = new Tensor([], [nbPatches, nbFilter])

    if (this._useWeblas && !(this._imColsMat._gpuMaxSizeExceeded || this._wRowsMat._gpuMaxSizeExceeded)) {
      // GPU
      const bias = this.bias ? this.weights.b.weblasTensor : this._zerosVec.weblasTensor
      matMul.tensor.data = weblas.pipeline.sgemm(
        1, this._imColsMat.weblasTensor, this._wRowsMat.weblasTensor,
        1, bias
      ).transfer()
    } else {
      // CPU
      if (this.bias) {
        for (let n = 0; n < nbFilter; n++) {
          ops.assigns(matMul.tensor.pick(null, n), this.weights.b.tensor.get(n))
        }
      }
      gemm(matMul.tensor, this._imColsMat.tensor, this._wRowsMat.tensor, 1, 1)
    }

    let output = new Tensor([], this.outputShape)
    let outputChannelRaveled = new Tensor([], [outputRows * outputCols])
    let outputChannel = new Tensor([], [outputRows, outputCols])
    for (let n = 0; n < nbFilter; n++) {
      ops.assign(outputChannelRaveled.tensor, matMul.tensor.pick(null, n))
      outputChannel.replaceTensorData(outputChannelRaveled.tensor.data)
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

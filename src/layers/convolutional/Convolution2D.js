import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'
import checkPipelineSupport from '../../utils/checkPipelineSupport'
import WebGLConv2D from '../../ext/convolutional/WebGLConv2D'

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

    this.activation = activation
    this.activationFunc = activations[activation]

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

    // Enable layer pipeline mode if supported
    if (this._useWeblas && this._pipelineEnabled) {
      const isPipelineModeSupported = checkPipelineSupport(this.layerClass, attrs)
      if (!isPipelineModeSupported) {
        this._pipelineEnabled = false
      } else {
        this.webglConv2D = new WebGLConv2D()
      }
    }
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
   * @param {number[]} inputShape
   */
  _calcOutputShape (inputShape) {
    const inputRows = inputShape[0]
    const inputCols = inputShape[1]
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
   * @param {number} [padValue]
   * @returns {Tensor} x
   */
  _padInput (x, padValue = 0) {
    if (this.borderMode === 'same') {
      const [inputRows, inputCols, inputChannels] = x.tensor.shape
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding
      const newRows = inputRows + paddingRowBefore + paddingRowAfter
      const newCols = inputCols + paddingColBefore + paddingColAfter
      let _x = new Tensor([], [newRows, newCols, inputChannels])
      if (padValue !== 0) {
        ops.assigns(_x.tensor, padValue)
      }
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
   * Convert input tensor to column matrix
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
   * Creates a index mapping from the 2D-tiled input tensor with associated
   * 3D tensor shape to the representation required prior to the matrix multiply.
   * This allows us to work directly on the 2D tiled tensor representations rather
   * than needing to reshape to the 3D reprentation and calling im2col.
   * @param {number[]} inputShape
   */
  _tiledIndexMapping (inputShape) {
    if (this._tiledIndexMappingRow && this._tiledIndexMappingCol) {
      return
    }

    let [inputRows, inputCols, inputChannels] = inputShape

    let indicesRow = new Tensor([], inputShape)
    let indicesCol = new Tensor([], inputShape)
    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        ops.assigns(indicesRow.tensor.pick(i, j, null), i * inputCols + j)
      }
    }
    for (let k = 0; k < inputChannels; k++) {
      ops.assigns(indicesCol.tensor.pick(null, null, k), k)
    }

    // padding for border mode 'same'
    if (this.borderMode === 'same') {
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding
      inputRows = inputRows + paddingRowBefore + paddingRowAfter
      inputCols = inputCols + paddingColBefore + paddingColAfter
      const padValue = -1
      this._padInput(indicesRow, padValue)
      this._padInput(indicesCol, padValue)
    }

    const nbRow = this.kernelShape[1]
    const nbCol = this.kernelShape[2]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const patchLen = nbRow * nbCol * inputChannels

    this._tiledIndexMappingRow = new Tensor([], [nbPatches, patchLen])
    this._tiledIndexMappingCol = new Tensor([], [nbPatches, patchLen])

    let patchRow = new Tensor([], [nbRow, nbCol, inputChannels])
    let patchCol = new Tensor([], [nbRow, nbCol, inputChannels])
    let offset = 0
    for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.subsample[0]) {
      for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.subsample[1]) {
        ops.assign(patchRow.tensor, indicesRow.tensor.hi(i + nbRow, j + nbCol, inputChannels).lo(i, j, 0))
        ops.assign(patchCol.tensor, indicesCol.tensor.hi(i + nbRow, j + nbCol, inputChannels).lo(i, j, 0))
        this._tiledIndexMappingRow.tensor.data.set(patchRow.tensor.data, offset)
        this._tiledIndexMappingCol.tensor.data.set(patchCol.tensor.data, offset)
        offset += patchLen
      }
    }
    this._tiledIndexMappingRow.createWeblasTensor()
    this._tiledIndexMappingCol.createWeblasTensor()
  }

  /**
   * Pipeline transfer
   * Typically called at the end of a pipelined layer sequence.

   * @param {Tensor} x
   * @returns {Tensor} x
   */
  transferFromPipeline (x) {
    if (!x.weblasTensor) {
      throw new Error('Variable passed in does not contain weblas tensor.')
    }

    const nbFilter = this.kernelShape[0]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols

    const tiled = new Tensor([], [nbPatches, nbFilter])
    tiled.tensor.data = x.weblasTensor.transfer()

    let output = new Tensor([], this.outputShape)
    let outputChannelRaveled = new Tensor([], [outputRows * outputCols])
    let outputChannel = new Tensor([], [outputRows, outputCols])
    for (let n = 0; n < nbFilter; n++) {
      ops.assign(outputChannelRaveled.tensor, tiled.tensor.pick(null, n))
      outputChannel.replaceTensorData(outputChannelRaveled.tensor.data)
      ops.assign(output.tensor.pick(null, null, n), outputChannel.tensor)
    }
    x.tensor = output.tensor

    return x
  }

  /**
   * Runs layer computational logic in pipeline mode
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callPipelineMode (x) {
    if (!x.weblasTensor) {
      throw new Error('Variable passed in does not contain weblas tensor.')
    }

    this._tiledIndexMapping(this.inputShape)

    // let test_in = new Tensor([], x.weblasTensor.shape)
    // test_in.tensor.data = x.weblasTensor.transfer(true)
    // let test_out = new Tensor([], this._tiledIndexMappingCol.tensor.shape)
    // for (let i = 0; i < this._tiledIndexMappingCol.tensor.shape[0]; i++) {
    //   for (let j = 0; j < this._tiledIndexMappingCol.tensor.shape[1]; j++) {
    //     test_out.tensor.set(i, j, test_in.tensor.get(
    //       this._tiledIndexMappingRow.tensor.get(i, j),
    //       this._tiledIndexMappingCol.tensor.get(i, j)
    //     ))
    //   }
    // }
    // console.log('pre',test_in.tensor.data.slice(20,30))
    // console.log('post', test_out.tensor.data.slice(20,30))

    const bias = this.bias ? this.weights.b.weblasTensor : this._zerosVec.weblasTensor
    x.weblasTensor = this.webglConv2D.call(
      x.weblasTensor,
      this._wRowsMat.weblasTensor,
      bias,
      this.activation,
      x._fromPipeline ? this._tiledIndexMappingRow.weblasTensor : null,
      x._fromPipeline ? this._tiledIndexMappingCol.weblasTensor : null
    )

    x._fromPipeline = true
    x._actualShape = this.outputShape

    return x
  }

  /**
   * Runs layer computational logic in regular mode
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callRegularMode (x) {
    if (!x.tensor) {
      throw new Error('Variable passed in does not contain tensor.')
    }

    // convert to tf ordering
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

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

    this.activationFunc(x)

    // convert back to th ordering if necessary
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(2, 0, 1)
    }

    return x
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    if (x._fromPipeline) {
      this.inputShape = x._actualShape
    } else {
      this.inputShape = x.tensor.shape
    }
    this._calcOutputShape(this.inputShape)

    if (this._pipelineEnabled) {
      if (!x._fromPipeline) {
        this._padInput(x)
        this._im2col(x)
        if (!this._imColsMat._gpuMaxSizeExceeded) {
          x.weblasTensor = this._imColsMat.weblasTensor
        } else {
          return this._callRegularMode(x)
        }
      }
      return this._callPipelineMode(x)
    } else {
      this._padInput(x)
      this._im2col(x)
      return this._callRegularMode(x)
    }
  }
}

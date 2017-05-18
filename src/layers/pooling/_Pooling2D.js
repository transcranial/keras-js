import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * _Pooling2D layer class
 */
export default class _Pooling2D extends Layer {
  /**
   * Creates a _Pooling2D layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = '_Pooling2D'

    const { pool_size = [2, 2], strides = null, padding = 'valid', data_format = 'channels_last' } = attrs

    if (Array.isArray(pool_size)) {
      this.poolSize = pool_size
    } else {
      this.poolSize = [pool_size, pool_size]
    }

    if (Array.isArray(strides)) {
      this.strides = strides
    } else if (strides !== null) {
      this.strides = [strides, strides]
    } else {
      this.strides = this.poolSize
    }

    this.padding = padding
    this.dataFormat = data_format

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max'
  }

  /**
   * Method for computing output dimensions and padding, based on input
   * dimensions, kernel size, and padding mode.
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   * @param {number[]} inputShape
   */
  _calcOutputShape(inputShape) {
    const [inputRows, inputCols, inputChannels] = inputShape
    const [nbRow, nbCol] = this.poolSize

    const outputRows = this.padding === 'same'
      ? Math.floor((inputRows + this.strides[0] - 1) / this.strides[0])
      : Math.floor((inputRows - nbRow + this.strides[0]) / this.strides[0])
    const outputCols = this.padding === 'same'
      ? Math.floor((inputCols + this.strides[1] - 1) / this.strides[1])
      : Math.floor((inputCols - nbCol + this.strides[1]) / this.strides[1])

    const paddingRow = this.padding === 'same'
      ? Math.max(0, Math.floor((outputRows - 1) * this.strides[0] + nbRow - inputRows))
      : 0
    const paddingCol = this.padding === 'same'
      ? Math.max(0, Math.floor((outputCols - 1) * this.strides[1] + nbCol - inputCols))
      : 0
    const paddingRowBefore = Math.floor(paddingRow / 2)
    const paddingRowAfter = paddingRow - paddingRowBefore
    const paddingColBefore = Math.floor(paddingCol / 2)
    const paddingColAfter = paddingCol - paddingColBefore

    this.outputShape = [outputRows, outputCols, inputChannels]
    this.inputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter]
  }

  /**
   * Pad input tensor if necessary, for padding='same'.
   * See above for notes on calculating padding.
   * For max, we pad with -infinity.
   * For average we pad with zero.
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _padInput(x) {
    if (this.padding === 'same') {
      const [inputRows, inputCols, inputChannels] = x.tensor.shape
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding
      const newRows = inputRows + paddingRowBefore + paddingRowAfter
      const newCols = inputCols + paddingColBefore + paddingColAfter

      let _x = new Tensor([], [newRows, newCols, inputChannels])
      if (this.poolingFunc === 'max') {
        ops.assigns(_x.tensor, Number.NEGATIVE_INFINITY)
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
   * Creates a index mapping from the 2D-tiled input tensor with associated
   * 3D tensor shape to the representation required prior to pooling.
   * @param {number[]} inputShape
   */
  _poolIndexMapping(inputShape) {
    if (this._poolIndicesPerChannel) {
      return
    }

    let inputRows = inputShape[0]
    let inputCols = inputShape[1]

    let indicesRow = new Tensor([], [inputRows, inputCols])
    let index = 0
    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        indicesRow.tensor.set(i, j, index)
        index += 1
      }
    }

    // padding for border mode 'same'
    if (this.padding === 'same') {
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding
      inputRows = inputRows + paddingRowBefore + paddingRowAfter
      inputCols = inputCols + paddingColBefore + paddingColAfter
      let _indicesRow = new Tensor([], [inputRows, inputCols])
      ops.assigns(_indicesRow.tensor, -1)
      ops.assign(
        _indicesRow.tensor
          .hi(inputShape[0] + paddingRowBefore, inputShape[1] + paddingColBefore)
          .lo(paddingRowBefore, paddingColBefore),
        indicesRow.tensor
      )
      indicesRow.tensor = _indicesRow.tensor
    }

    const [nbRow, nbCol] = this.poolSize
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]

    this._poolIndicesPerChannel = new Tensor([], [outputRows * outputCols, nbRow * nbCol])

    let patchRow = new Tensor([], [nbRow, nbCol])
    let offset = 0
    for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.strides[1]) {
        ops.assign(patchRow.tensor, indicesRow.tensor.hi(i + nbRow, j + nbCol).lo(i, j))
        this._poolIndicesPerChannel.tensor.data.set(patchRow.tensor.data, offset)
        offset += nbRow * nbCol
      }
    }
    this._poolIndicesPerChannel.createWeblasTensor()
  }

  /**
   * Runs layer computational logic in pipeline mode
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callPipelineMode(x) {
    if (!x.weblasTensor) {
      throw new Error('Variable passed in does not contain weblas tensor.')
    }

    this._calcOutputShape(x._actualShape)
    this._poolIndexMapping(x._actualShape)

    x.weblasTensor = this.webglPooling2D.call(x.weblasTensor, this._poolIndicesPerChannel.weblasTensor)

    x._fromPipeline = true
    x._actualShape = this.outputShape

    return x
  }

  /**
   * Runs layer computational logic in regular mode
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callRegularMode(x) {
    if (!x.tensor) {
      throw new Error('Variable passed in does not contain tensor.')
    }

    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

    this._calcOutputShape(x.tensor.shape)
    this._padInput(x)

    const [inputRows, inputCols, inputChannels] = x.tensor.shape
    const [nbRow, nbCol] = this.poolSize
    let y = new Tensor([], this.outputShape)
    let patch = new Tensor([], [nbRow, nbCol, inputChannels])

    // keep track of padding since these values are not included in pooling
    // for max, we can ignore since padding values are set to -infinity
    const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding

    for (let i = 0, _i = 0; i <= inputRows - nbRow; (i += this.strides[0]), _i++) {
      let nbRowInPadding = 0
      if (i < paddingRowBefore) {
        nbRowInPadding = paddingRowBefore - i
      } else if (i + nbRow > inputRows - paddingRowAfter) {
        nbRowInPadding = i + nbRow - (inputRows - paddingRowAfter)
      }

      for (let j = 0, _j = 0; j <= inputCols - nbCol; (j += this.strides[1]), _j++) {
        let nbColInPadding = 0
        if (j < paddingColBefore) {
          nbColInPadding = paddingColBefore - j
        } else if (j + nbCol > inputCols - paddingColAfter) {
          nbColInPadding = j + nbCol - (inputCols - paddingColAfter)
        }

        ops.assign(patch.tensor, x.tensor.hi(i + nbRow, j + nbCol, inputChannels).lo(i, j, 0))
        for (let c = 0; c < inputChannels; c++) {
          if (this.poolingFunc === 'max') {
            y.tensor.set(_i, _j, c, ops.sup(patch.tensor.pick(null, null, c)))
          } else if (this.poolingFunc === 'average') {
            let nbCellsEffective = (nbRow - nbRowInPadding) * (nbCol - nbColInPadding)
            y.tensor.set(_i, _j, c, ops.sum(patch.tensor.pick(null, null, c)) / nbCellsEffective)
          }
        }
      }
    }

    x.tensor = y.tensor

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(2, 0, 1)
    }

    return x
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    if (this._pipelineEnabled && x._fromPipeline) {
      return this._callPipelineMode(x)
    } else {
      return this._callRegularMode(x)
    }
  }
}

import Tensor from '../../Tensor'
import Convolution2D from './Convolution2D'
import ops from 'ndarray-ops'
import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'

/**
 * AtrousConvolution2D layer class
 * This class extends the Convolution2D layer class, and overrides the methods
 * `_calcOutputShape` and `_im2col` by creating filter dilations based on the
 * specified `atrousRate`.
 */
export default class AtrousConvolution2D extends Convolution2D {
  /**
   * Creates a AtrousConvolution2D layer
   * @param {number} nbFilter - Number of convolution filters to use.
   * @param {number} nbRow - Number of rows in the convolution kernel.
   * @param {number} nbCol - Number of columns in the convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor (nbFilter, nbRow, nbCol, attrs = {}) {
    super(nbFilter, nbRow, nbCol, attrs)
    const {
      atrousRate = [1, 1]
    } = attrs
    this.atrousRate = atrousRate
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
}

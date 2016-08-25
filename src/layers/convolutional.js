import * as activations from '../activations'
import Tensor from '../tensor'
import { Layer } from '../engine/topology'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'
import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'

/**
* Convolution2D layer class
*/
export class Convolution2D extends Layer {
  /**
  * Creates a Convolution2D layer
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
      dimOrdering = 'tf',
      bias = true
    } = attrs

    this.kernelShape = [nbFilter, nbRow, nbCol]

    this.activation = activations[activation]

    if (['valid', 'same'].indexOf(borderMode) > -1) {
      this.borderMode = borderMode
    } else {
      throw new Error(`${this.name} [Convolution2D layer] Invalid borderMode.`)
    }

    this.subsample = subsample

    if (['tf', 'th'].indexOf(dimOrdering) > -1) {
      this.dimOrdering = dimOrdering
    } else {
      throw new Error(`${this.name} [Convolution2D layer] Invalid dimOrdering.`)
    }

    this.bias = bias

    /**
    * Layer weights specification
    */
    this.params = this.bias ? ['W', 'b'] : ['W']
  }

  /**
  * Method for computing output dimensions based on input dimensions and kernel size
  * @param {Tensor} x
  * @returns {number[]} [outputRows, outputCols, outputChannels]
  */
  _calcOutputShape = x => {
    const inputRows = x.tensor.shape[0]
    const inputCols = x.tensor.shape[1]
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    const paddingRow = this.borderMode === 'same' ? Math.floor(nbRow / 2) : 0
    const paddingCol = this.borderMode === 'same' ? Math.floor(nbCol / 2) : 0
    const outputRows = (inputRows + 2 * paddingRow - nbRow) / this.subsample[0] + 1
    const outputCols = (inputCols + 2 * paddingCol - nbCol) / this.subsample[1] + 1
    const outputChannels = nbFilter
    this.outputShape = [outputRows, outputCols, outputChannels]
  }

  /**
  * image to column matrix
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  _im2col = x => {
    const inputChannels = x.tensor.shape[2]
    const nbRow = this.kernelShape[1]
    const nbCol = this.kernelShape[2]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const patchLen = nbRow * nbCol * inputChannels

    const imColsMat = new Tensor([], [patchLen, nbPatches])

    let patch = new Tensor([], [patchLen])
    let n = 0
    for (let i = 0; i < outputRows; i += this.subsample[0]) {
      for (let j = 0; j < outputCols; j += this.subsample[1]) {
        const patchData = flattenDeep(unpack(
          x.tensor.hi(i + nbRow, j + nbCol, inputChannels).lo(i, j, 0)
        ))
        patch.replaceTensorData(patchData)
        ops.assign(imColsMat.tensor.pick(null, n), patch.tensor)
        n += 1
      }
    }

    return imColsMat
  }

  /**
  * filters to row matrix
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  _w2row = x => {
    const inputChannels = x.tensor.shape[2]
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    const patchLen = nbRow * nbCol * inputChannels

    const wRowsMat = new Tensor([], [nbFilter, patchLen])

    let patch = new Tensor([], [patchLen])
    for (let n = 0; n < nbFilter; n++) {
      const patchData = flattenDeep(unpack(
        this.weights.W.tensor.pick(null, null, null, n)
      ))
      patch.replaceTensorData(patchData)
      ops.assign(wRowsMat.tensor.pick(n, null), patch.tensor)
    }

    return wRowsMat
  }

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    this._calcOutputShape(x)

    const imColsMat = this._im2col(x)
    const wRowsMat = this._w2row(x)

    const nbFilter = this.kernelShape[0]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const matMul = new Tensor([], [nbFilter, nbPatches])
    if (this.bias) {
      for (let n = 0; n < nbFilter; n++) {
        ops.assigns(matMul.tensor.pick(n, null), this.weights.b.tensor.get(n))
      }
    }
    gemm(matMul.tensor, wRowsMat.tensor, imColsMat.tensor, 1, 1)

    let output = new Tensor([], this.outputShape)
    let outputChannel = new Tensor([], [outputRows, outputCols])
    for (let n = 0; n < nbFilter; n++) {
      const outputChannelData = flattenDeep(unpack(
        matMul.tensor.pick(n, null)
      ))
      outputChannel.replaceTensorData(outputChannelData)
      ops.assign(output.tensor.pick(null, null, n), outputChannel.tensor)
    }
    x.tensor = output.tensor

    this.activation(x)

    return x
  }
}

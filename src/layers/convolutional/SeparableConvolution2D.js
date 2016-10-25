import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import Convolution2D from './Convolution2D'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'

/**
 * _DepthwiseConvolution2D layer class
 */
class _DepthwiseConvolution2D extends Convolution2D {
  constructor (attrs = {}) {
    super(attrs)
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
    const patchLen = nbRow * nbCol

    if (!this._imColsMat) {
      this._imColsMat = new Tensor([], [nbPatches * inputChannels, patchLen])
    }

    let patch = new Tensor([], [nbRow, nbCol, 1])
    let offset = 0
    for (let c = 0; c < inputChannels; c++) {
      for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.subsample[0]) {
        for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.subsample[1]) {
          ops.assign(patch.tensor, x.tensor.hi(i + nbRow, j + nbCol, c + 1).lo(i, j, c))
          this._imColsMat.tensor.data.set(patch.tensor.data, offset)
          offset += patchLen
        }
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
    const patchLen = nbRow * nbCol

    this._wRowsMat = new Tensor([], [patchLen, nbFilter * inputChannels])

    let patch = new Tensor([], [nbRow, nbCol])
    let patchRaveled = new Tensor([], [patchLen])
    let p = 0
    for (let c = 0; c < inputChannels; c++) {
      for (let n = 0; n < nbFilter; n++) {
        ops.assign(patch.tensor, this.weights.W.tensor.pick(null, null, c, n))
        patchRaveled.replaceTensorData(patch.tensor.data)
        ops.assign(this._wRowsMat.tensor.pick(null, p), patchRaveled.tensor)
        p += 1
      }
    }

    return this._wRowsMat
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    let startTime = performance.now()
    this._calcOutputShape(x)
    let endTime = performance.now()
    console.log(0, endTime - startTime)
    startTime = performance.now()
    this._padInput(x)
    endTime = performance.now()
    console.log(1, endTime - startTime)

    startTime = performance.now()
    this._im2col(x)
    endTime = performance.now()
    console.log(2, endTime - startTime)

    startTime = performance.now()
    const nbFilter = this.kernelShape[0]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const matMul = new Tensor([], [nbPatches * x.tensor.shape[2], nbFilter * x.tensor.shape[2]])
    endTime = performance.now()
    console.log(3, endTime - startTime)

    startTime = performance.now()
    if (this._useWeblas) {
      // GPU
      if (this._imColsMat.weblasTensorsSplit) {
        // split matrix multiply if this._imColsMat dimension > webgl.MAX_TEXTURE_SIZE
        let offset = 0
        this._imColsMat.weblasTensorsSplit.forEach(imColsMatSplit => {
          const matMulSplitData = weblas.pipeline.sgemm(
            1, imColsMatSplit, this._wRowsMat.weblasTensor,
            1, this._zerosVec.weblasTensor
          ).transfer()
          matMul.tensor.data.set(matMulSplitData, offset)
          offset += matMulSplitData.length
        })
      } else {
        // normal matrix multiply
        matMul.tensor.data = weblas.pipeline.sgemm(
          1, this._imColsMat.weblasTensor, this._wRowsMat.weblasTensor,
          1, this._zerosVec.weblasTensor
        ).transfer()
      }
    } else {
      // CPU
      gemm(matMul.tensor, this._imColsMat.tensor, this._wRowsMat.tensor, 1, 1)
    }
    endTime = performance.now()
    console.log(4, endTime - startTime)

    startTime = performance.now()
    let output = new Tensor([], [outputRows, outputCols, x.tensor.shape[2] * nbFilter])
    const outputDataLength = outputRows * outputCols * x.tensor.shape[2] * nbFilter
    let dataFiltered = new Float32Array(outputDataLength)
    for (let c = 0; c < x.tensor.shape[2]; c++) {
      for (let i = 0, n = c * outputDataLength + c * nbFilter, len = (c + 1) * outputDataLength; n < len; i++, n += nbFilter * x.tensor.shape[2]) {
        for (let m = 0; m < nbFilter; m++) {
          dataFiltered[n + m - c * outputDataLength] = matMul.tensor.data[n + m]
        }
      }
    }
    output.replaceTensorData(dataFiltered)
    endTime = performance.now()
    console.log(5, endTime - startTime)

    x.tensor = output.tensor

    return x
  }
}

/**
 * SeparableConvolution2D layer class
 */
export default class SeparableConvolution2D extends Layer {
  /**
   * Creates a SeparableConvolution2D layer
   * @param {number} attrs.nbFilter - Number of convolution filters to use.
   * @param {number} attrs.nbRow - Number of rows in the convolution kernel.
   * @param {number} attrs.nbCol - Number of columns in the convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor (attrs = {}) {
    super(attrs)
    this.layerClass = 'SeparableConvolution2D'

    const {
      nbFilter = 1,
      nbRow = 1,
      nbCol = 1,
      activation = 'linear',
      borderMode = 'valid',
      subsample = [1, 1],
      depthMultiplier = 1,
      dimOrdering = 'tf',
      bias = true
    } = attrs

    this.activation = activations[activation]

    if (borderMode === 'valid' || borderMode === 'same') {
      this.borderMode = borderMode
    } else {
      throw new Error(`${this.name} [SeparableConvolution2D layer] Invalid borderMode.`)
    }

    this.subsample = subsample
    this.depthMultiplier = depthMultiplier

    if (dimOrdering === 'tf' || dimOrdering === 'th') {
      this.dimOrdering = dimOrdering
    } else {
      throw new Error(`${this.name} [SeparableConvolution2D layer] Only tf and th dim ordering are allowed.`)
    }

    this.bias = bias

    // Layer weights specification
    this.params = this.bias
      ? ['depthwise_kernel', 'pointwise_kernel', 'b']
      : ['depthwise_kernel', 'pointwise_kernel']

    // SeparableConvolution2D has two components: depthwise, and pointwise.
    // Activation function and bias is applied at the end.
    // Subsampling (striding) only performed on depthwise part, not the pointwise part.
    this.depthwiseConvAttrs = { nbFilter: this.depthMultiplier, nbRow, nbCol, activation: 'linear', borderMode, subsample, dimOrdering, bias: false, gpu: attrs.gpu }
    this.pointwiseConvAttrs = { nbFilter, nbRow: 1, nbCol: 1, activation: 'linear', borderMode, subsample: [1, 1], dimOrdering, bias: this.bias, gpu: attrs.gpu }
  }

  /**
   * Method for setting layer weights
   * Override `super` method since weights must be set in component Convolution2D layers
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights (weightsArr) {
    this._depthwiseConv = new _DepthwiseConvolution2D(this.depthwiseConvAttrs)
    this._depthwiseConv.setWeights(weightsArr.slice(0, 1))
    this._pointwiseConv = new Convolution2D(this.pointwiseConvAttrs)
    this._pointwiseConv.setWeights(weightsArr.slice(1, 3))
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    // Perform depthwise ops
    const depthwiseOutput = this._depthwiseConv.call(x)

    // Perform depthwise ops
    let startTime = performance.now()
    const pointwiseOutput = this._pointwiseConv.call(depthwiseOutput)
    let endTime = performance.now()
    console.log(6, endTime - startTime)

    x.tensor = pointwiseOutput.tensor

    // activation
    this.activation(x)

    return x
  }
}

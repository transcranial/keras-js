import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import Conv2D from './Conv2D'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'

/**
 * _DepthwiseConv2D layer class
 */
class _DepthwiseConv2D extends Conv2D {
  constructor(attrs = {}) {
    super(attrs)
  }

  /**
   * Convert input image to column matrix
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _im2col(x) {
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
      for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.strides[0]) {
        for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.strides[1]) {
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
  _w2row() {
    const inputChannels = this.weights['kernel'].tensor.shape[2]
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    const patchLen = nbRow * nbCol

    this._wRowsMat = new Tensor([], [patchLen, nbFilter * inputChannels])

    let patch = new Tensor([], [nbRow, nbCol])
    let patchRaveled = new Tensor([], [patchLen])
    let p = 0
    for (let c = 0; c < inputChannels; c++) {
      for (let n = 0; n < nbFilter; n++) {
        ops.assign(patch.tensor, this.weights['kernel'].tensor.pick(null, null, c, n))
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
  call(x) {
    this._calcOutputShape(x.tensor.shape)
    this._padInput(x)

    this._im2col(x)

    const nbFilter = this.kernelShape[0]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const matMul = new Tensor([], [nbPatches * x.tensor.shape[2], nbFilter * x.tensor.shape[2]])

    if (this._useWeblas && !(this._imColsMat._gpuMaxSizeExceeded || this._wRowsMat._gpuMaxSizeExceeded)) {
      // GPU
      matMul.tensor.data = weblas.pipeline
        .sgemm(1, this._imColsMat.weblasTensor, this._wRowsMat.weblasTensor, 1, this._zerosVec.weblasTensor)
        .transfer()
    } else {
      // CPU
      gemm(matMul.tensor, this._imColsMat.tensor, this._wRowsMat.tensor, 1, 1)
    }

    let output = new Tensor([], [outputRows, outputCols, x.tensor.shape[2] * nbFilter])
    const outputDataLength = outputRows * outputCols * x.tensor.shape[2] * nbFilter
    let dataFiltered = new Float32Array(outputDataLength)
    for (let c = 0; c < x.tensor.shape[2]; c++) {
      for (
        let i = 0, n = c * outputDataLength + c * nbFilter, len = (c + 1) * outputDataLength;
        n < len;
        i++, (n += nbFilter * x.tensor.shape[2])
      ) {
        for (let m = 0; m < nbFilter; m++) {
          dataFiltered[n + m - c * outputDataLength] = matMul.tensor.data[n + m]
        }
      }
    }
    output.replaceTensorData(dataFiltered)

    x.tensor = output.tensor

    return x
  }
}

/**
 * SeparableConv2D layer class
 */
export default class SeparableConv2D extends Layer {
  /**
   * Creates a SeparableConv2D layer
   * @param {Number} attrs.filters - Number of convolution filters to use.
   * @param {Array<Number>|Number} attrs.kernel_size - Size of the convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'SeparableConv2D'

    const {
      filters = 1,
      kernel_size = [1, 1],
      strides = [1, 1],
      padding = 'valid',
      data_format = 'channels_last',
      depth_multiplier = 1,
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
      throw new Error(`${this.name} [Conv2D layer] Invalid padding.`)
    }

    if (data_format === 'channels_last' || data_format === 'channels_first') {
      this.dataFormat = data_format
    } else {
      throw new Error(`${this.name} [Conv2D layer] Only channels_last and channels_first data formats are allowed.`)
    }

    this.activation = activation
    this.activationFunc = activations[activation]

    if (padding === 'valid' || padding === 'same') {
      this.padding = padding
    } else {
      throw new Error(`${this.name} [SeparableConv2D layer] Invalid padding.`)
    }

    this.use_bias = use_bias

    // Layer weights specification
    this.params = this.use_bias
      ? ['depthwise_kernel', 'pointwise_kernel', 'bias']
      : ['depthwise_kernel', 'pointwise_kernel']

    // SeparableConv2D has two components: depthwise, and pointwise.
    // Activation function and bias is applied at the end.
    // Subsampling (striding) only performed on depthwise part, not the pointwise part.
    this.depthwiseConvAttrs = {
      filters: depth_multiplier,
      kernel_size: [this.kernelShape[1], this.kernelShape[2]],
      strides: this.strides,
      padding,
      data_format,
      activation: 'linear',
      use_bias: false,
      gpu: attrs.gpu
    }
    this.pointwiseConvAttrs = {
      filters,
      kernel_size: [1, 1],
      strides: [1, 1],
      padding,
      data_format,
      activation: 'linear',
      use_bias,
      gpu: attrs.gpu
    }
  }

  /**
   * Method for setting layer weights
   * Override `super` method since weights must be set in component Conv2D layers
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    this._depthwiseConv = new _DepthwiseConv2D(this.depthwiseConvAttrs)
    this._depthwiseConv.setWeights(weightsArr.slice(0, 1))
    this._pointwiseConv = new Conv2D(this.pointwiseConvAttrs)
    this._pointwiseConv.setWeights(weightsArr.slice(1, 3))
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    // Perform depthwise ops
    const depthwiseOutput = this._depthwiseConv.call(x)

    // Perform depthwise ops
    const pointwiseOutput = this._pointwiseConv.call(depthwiseOutput)

    x.tensor = pointwiseOutput.tensor

    // activation
    this.activationFunc(x)

    return x
  }
}

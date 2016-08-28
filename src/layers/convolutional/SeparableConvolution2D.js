import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../engine/Layer'
import Convolution2D from './Convolution2D'
import ops from 'ndarray-ops'

/**
 * SeparableConvolution2D layer class
 */
export default class SeparableConvolution2D extends Layer {
  /**
   * Creates a SeparableConvolution2D layer
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
    const depthwiseConvAttrs = { activation: 'linear', borderMode, subsample, dimOrdering, bias: false }
    const pointwiseConvAttrs = { activation: 'linear', borderMode, subsample: [1, 1], dimOrdering, bias: false }
    this._depthwiseConv = new Convolution2D(this.depthMultiplier, nbRow, nbCol, depthwiseConvAttrs)
    this._pointwiseConv = new Convolution2D(nbFilter, 1, 1, pointwiseConvAttrs)
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    // Perform depthwise ops
    this._depthwiseConv._calcOutputShape(x)
    const outputRows = this._depthwiseConv.outputShape[0]
    const outputCols = this._depthwiseConv.outputShape[1]
    let depthwiseOutput = new Tensor([], [outputRows, outputCols, this.depthMultiplier])

    // temporary tensor to hold input for a particular channel
    const [inputRows, inputCols, inputChannels] = x.tensor.shape
    let inputChannelSlice = new Tensor([], [inputRows, inputCols, 1])

    // temporary tensor to hold weights for a particular channel
    let depthwiseKernelSliceShape = this.weights.depthwise_kernel.tensor.shape.slice()
    depthwiseKernelSliceShape[2] = 1
    let depthwiseKernelSlice = new Tensor([], depthwiseKernelSliceShape)

    // tensor to hold combined output of depthwise ops
    const depthwiseOutputCombined = new Tensor([], [
      this._depthwiseConv.outputShape[0],
      this._depthwiseConv.outputShape[1],
      inputChannels * this.depthMultiplier
    ])

    // perform convolution over each channel separately
    for (let c = 0; c < inputChannels; c++) {
      depthwiseKernelSlice.tensor = this.weights.depthwise_kernel.tensor
        .hi(depthwiseKernelSliceShape[0], depthwiseKernelSliceShape[1], c + 1, depthwiseKernelSliceShape[3])
        .lo(0, 0, c, 0)
      this._depthwiseConv.setWeights([depthwiseKernelSlice])
      inputChannelSlice.tensor = x.tensor.hi(inputRows, inputCols, c + 1).lo(0, 0, c)
      depthwiseOutput = this._depthwiseConv.call(inputChannelSlice)
      ops.assign(
        depthwiseOutputCombined.tensor
          .hi(outputRows, outputCols, (c + 1) * this.depthMultiplier)
          .lo(0, 0, c * this.depthMultiplier),
        depthwiseOutput.tensor
      )
    }

    // Perform depthwise ops
    this._pointwiseConv.setWeights([this.weights.pointwise_kernel])
    const pointwiseOutput = this._pointwiseConv.call(depthwiseOutputCombined)

    // bias
    if (this.bias) {
      for (let n = 0; n < this.weights.b.tensor.shape[0]; n++) {
        ops.addseq(pointwiseOutput.tensor.pick(null, null, n), this.weights.b.tensor.get(n))
      }
    }
    x.tensor = pointwiseOutput.tensor

    // activation
    this.activation(x)

    return x
  }
}

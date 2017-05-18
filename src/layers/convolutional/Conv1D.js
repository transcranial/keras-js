import Layer from '../../Layer'
import Conv2D from './Conv2D'
import squeeze from 'ndarray-squeeze'
import unsqueeze from 'ndarray-unsqueeze'

/**
 * Conv1D layer class
 */
export default class Conv1D extends Layer {
  /**
   * Creates a Conv1D layer
   * @param {Number} attrs.filters - Number of convolution filters to use.
   * @param {Number} attrs.kernel_size - Length of 1D convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Conv1D'

    const {
      filters = 1,
      kernel_size = 1,
      strides = 1,
      padding = 'valid',
      dilation_rate = 1,
      activation = 'linear',
      use_bias = true
    } = attrs

    if (padding !== 'valid' && padding !== 'same') {
      throw new Error(`${this.name} [Conv1D layer] Invalid padding.`)
    }

    if (dilation_rate !== 1 && strides !== 1) {
      // Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1
      // https://keras.io/layers/convolutional/#conv1d
      throw new Error(`${this.name} [Conv1D layer] Incompatible combination of dilation_rate with strides.`)
    }

    this.use_bias = use_bias

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel']

    // Bootstrap Conv2D layer:
    // Conv1D is actually a shim on top of Conv2D, where
    // all of the computational action is performed
    // Note that we use `channels_first` dim ordering here.
    const conv2dAttrs = {
      filters,
      kernel_size: [kernel_size, 1],
      strides: [strides, 1],
      padding,
      data_format: 'channels_first',
      dilation_rate,
      activation,
      use_bias
    }
    this._conv2dAttrs = conv2dAttrs
    this._conv2d = new Conv2D(Object.assign(conv2dAttrs, { gpu: attrs.gpu }))
  }

  /**
   * Method for setting layer weights
   * Override `super` method since weights must be set in `this._conv2d`
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    weightsArr[0].tensor = unsqueeze(weightsArr[0].tensor).transpose(2, 1, 0, 3)
    this._conv2d.setWeights(weightsArr)
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    x.tensor = unsqueeze(x.tensor).transpose(0, 2, 1)
    const conv2dOutput = this._conv2d.call(x)
    x.tensor = squeeze(conv2dOutput.tensor).transpose(1, 0, 2)
    return x
  }
}

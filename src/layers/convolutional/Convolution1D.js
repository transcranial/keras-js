import Layer from '../../Layer'
import Convolution2D from './Convolution2D'
import squeeze from 'ndarray-squeeze'
import unsqueeze from 'ndarray-unsqueeze'

/**
 * Convolution1D layer class
 */
export default class Convolution1D extends Layer {
  /**
   * Creates a Convolution1D layer
   * @param {number} attrs.nbFilter - Number of convolution filters to use.
   * @param {number} attrs.filterLength - Length of 1D convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor (attrs = {}) {
    super(attrs)
    const {
      nbFilter = 1,
      filterLength = 1,
      activation = 'linear',
      borderMode = 'valid',
      subsampleLength = 1,
      bias = true
    } = attrs

    if (borderMode !== 'valid' && borderMode !== 'same') {
      throw new Error(`${this.name} [Convolution1D layer] Invalid borderMode.`)
    }

    // Layer weights specification
    this.params = this.bias ? ['W', 'b'] : ['W']

    // Bootstrap Convolution2D layer:
    // Convolution1D is actually a shim on top of Convolution2D, where
    // all of the computational action is performed
    // Note that Keras uses `th` dim ordering here.
    this._conv2d = new Convolution2D({
      nbFilter,
      nbRow: filterLength,
      nbCol: 1,
      activation,
      borderMode,
      subsample: [subsampleLength, 1],
      dimOrdering: 'th',
      bias
    })
  }

  /**
   * Method for setting layer weights
   * Override `super` method since weights must be set in `this._conv2d`
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights (weightsArr) {
    this._conv2d.setWeights(weightsArr)
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    x.tensor = unsqueeze(x.tensor).transpose(1, 0, 2)
    const conv2dOutput = this._conv2d.call(x)
    x.tensor = squeeze(conv2dOutput.tensor).transpose(1, 0, 2)
    return x
  }
}

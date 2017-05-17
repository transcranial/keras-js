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
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Convolution1D'

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

    this.bias = bias

    // Layer weights specification
    this.params = this.bias ? ['W', 'b'] : ['W']

    // Bootstrap Convolution2D layer:
    // Convolution1D is actually a shim on top of Convolution2D, where
    // all of the computational action is performed
    // Note that Keras uses `th` dim ordering here.
    const conv2dAttrs = {
      nbFilter,
      nbRow: filterLength,
      nbCol: 1,
      activation,
      borderMode,
      subsample: [subsampleLength, 1],
      dimOrdering: 'th',
      bias
    }
    this._conv2dAttrs = conv2dAttrs
    this._conv2d = new Convolution2D(Object.assign(conv2dAttrs, { gpu: attrs.gpu }))
  }

  /**
   * Method for setting layer weights
   * Override `super` method since weights must be set in `this._conv2d`
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    const { nbFilter, nbRow, nbCol } = this._conv2dAttrs
    let shape = weightsArr[0].tensor.shape

    // check for legacy shape of weights
    // Keras:    (nb_filter, input_dim, filter_length, 1)
    // Keras.js: (nbFilter, inputChannels, nbRow, nbCol)
    if (!(shape[0] === nbRow && shape[1] === nbCol) || shape[3] !== nbFilter) {
      console.warn('Using legacy shape of weights')

      if (!((shape[0] === nbFilter) & ((shape[2] === nbRow) & (shape[3] === nbCol)))) {
        throw new Error('Unsupported shape of weights')
      }
    } else {
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(3, 2, 0, 1)
    }
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

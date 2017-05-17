import Layer from '../../Layer'

/**
 * GaussianNoise layer class
 * Note that this layer is here only for compatibility purposes,
 * as it's only active during training phase.
 */
export default class GaussianNoise extends Layer {
  /**
   * Creates a GaussianNoise layer
   * @param {number} attrs.p - fraction of the input units to drop (between 0 and 1)
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'GaussianNoise'

    const { sigma = 0 } = attrs
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    return x
  }
}

import Layer from '../../Layer'

/**
 * Dropout layer class
 * Note that this layer is here only for compatibility purposes,
 * as it's only active during training phase.
 */
export default class Dropout extends Layer {
  /**
   * Creates an Dropout layer
   * @param {number} attrs.p - fraction of the input units to drop (between 0 and 1)
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Dropout'

    const { p = 0.5 } = attrs

    this.p = Math.min(Math.max(0, p), 1)
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

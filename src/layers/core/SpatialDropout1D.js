import Layer from '../../Layer'

/**
 * SpatialDropout1D layer class
 * Note that this layer is here only for compatibility purposes,
 * as it's only active during training phase.
 */
export default class SpatialDropout1D extends Layer {
  /**
   * Creates an SpatialDropout1D layer
   * @param {number} attrs.p - fraction of the input units to drop (between 0 and 1)
   * @param {number} [attrs.dimOrdering] - `tf` or `th`
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'SpatialDropout1D'

    const { p = 0.5, dimOrdering = 'tf' } = attrs

    this.p = Math.min(Math.max(0, p), 1)
    this.dimOrdering = dimOrdering
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

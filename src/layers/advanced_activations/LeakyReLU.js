import Layer from '../../Layer'
import { relu } from '../../activations'

/**
 * LeakyReLU advanced activation layer class
 */
export default class LeakyReLU extends Layer {
  /**
   * Creates a LeakyReLU activation layer
   * @param {number} attrs.alpha - negative slope coefficient
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'LeakyReLU'

    const { alpha = 0.3 } = attrs

    this.alpha = alpha
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    relu(x, { alpha: this.alpha })
    return x
  }
}

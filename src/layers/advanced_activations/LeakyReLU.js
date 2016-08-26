import Layer from '../../engine/Layer'
import { relu } from '../../activations'

/**
 * LeakyReLU advanced activation layer class
 */
export default class LeakyReLU extends Layer {
  /**
   * Creates a LeakyReLU activation layer
   * @param {number} alpha - negative slope coefficient
   */
  constructor (alpha = 0.3) {
    super({})
    this.alpha = alpha
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call = x => {
    relu(x, { alpha: this.alpha })
    return x
  }
}

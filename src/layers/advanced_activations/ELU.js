import Layer from '../../Layer'
import cwise from 'cwise'

/**
 * ELU advanced activation layer class
 */
export default class ELU extends Layer {
  /**
   * Creates a ELU activation layer
   * @param {number} attrs.alpha - scale for the negative factor
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'ELU'

    const { alpha = 1.0 } = attrs

    this.alpha = alpha
  }

  _compute = cwise({
    args: ['array', 'scalar'],
    body: function(_x, alpha) {
      _x = Math.max(_x, 0) + alpha * (Math.exp(Math.min(_x, 0)) - 1)
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    this._compute(x.tensor, this.alpha)
    return x
  }
}

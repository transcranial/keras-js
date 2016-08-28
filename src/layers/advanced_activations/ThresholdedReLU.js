import Layer from '../../engine/Layer'
import cwise from 'cwise'

/**
 * ThresholdedReLU advanced activation layer class
 */
export default class ThresholdedReLU extends Layer {
  /**
   * Creates a ThresholdedReLU activation layer
   * @param {number} theta - float >= 0. Threshold location of activation.
   */
  constructor (theta = 1.0, attrs = {}) {
    super(attrs)
    this.theta = theta
  }

  _compute = cwise({
    args: ['array', 'scalar'],
    body: function (_x, theta) {
      _x = _x * Number(_x > theta)
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    this._compute(x.tensor, this.theta)
    return x
  }
}

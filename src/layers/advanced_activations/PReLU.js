import Layer from '../../Layer'
import cwise from 'cwise'

/**
 * PReLU advanced activation layer class
 * reference code:
 * ```
 * pos = K.relu(x)
 * neg = self.alphas * (x - abs(x)) * 0.5
 * return pos + neg
 * ```
 */
export default class PReLU extends Layer {
  /**
   * Creates a PReLU activation layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'PReLU'

    // Layer weights specification
    this.params = ['alphas']
  }

  _compute = cwise({
    args: ['array', 'array'],
    body: function(_x, alpha) {
      _x = Math.max(_x, 0) + alpha * Math.min(_x, 0)
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    this._compute(x.tensor, this.weights.alphas.tensor)
    return x
  }
}

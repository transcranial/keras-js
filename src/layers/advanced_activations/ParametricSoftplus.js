import Layer from '../../Layer'
import cwise from 'cwise'

/**
 * ParametricSoftplus advanced activation layer class
 * alpha * log(1 + exp(beta * X))
 */
export default class ParametricSoftplus extends Layer {
  /**
   * Creates a ParametricSoftplus activation layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'ParametricSoftplus'

    // Layer weights specification
    this.params = ['alphas', 'betas']
  }

  _compute = cwise({
    args: ['array', 'array', 'array'],
    body: function(_x, alpha, beta) {
      _x = alpha * Math.log(1 + Math.exp(beta * _x))
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    this._compute(x.tensor, this.weights.alphas.tensor, this.weights.betas.tensor)
    return x
  }
}

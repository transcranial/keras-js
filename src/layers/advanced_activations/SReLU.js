import Layer from '../../Layer'
import cwise from 'cwise'

/**
 * SReLU advanced activation layer class
 * S-shaped Rectified Linear Unit
 */
export default class SReLU extends Layer {
  /**
   * Creates a SReLU activation layer
   */
  constructor (attrs = {}) {
    super(attrs)

    // Layer weights specification
    this.params = ['t_left', 'a_left', 't_right', 'a_right']
  }

  // t_right_actual = t_left + abs(t_right)
  // Y_left_and_center = t_left + K.relu(x - t_left, a_left, t_right_actual - t_left)
  // Y_right = K.relu(x - t_right_actual) * a_right
  // return Y_left_and_center + Y_right
  _compute = cwise({
    args: ['array', 'array', 'array', 'array', 'array'],
    body: function (_x, tL, aL, tR, aR) {
      _x = tL + Math.min(Math.max(_x - tL, 0), Math.abs(tR)) + aL * Math.min(_x - tL, 0) +
        Math.max(_x - (tL + Math.abs(tR)), 0) * aR
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    this._compute(
      x.tensor,
      this.weights.t_left.tensor,
      this.weights.a_left.tensor,
      this.weights.t_right.tensor,
      this.weights.a_right.tensor
    )
    return x
  }
}

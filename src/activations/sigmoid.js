import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from '../Tensor'

const _sigmoid = cwise({
  args: ['array'],
  body: function(_x) {
    _x = 1 / (1 + Math.exp(-_x))
  }
})

/**
 * Sigmoid activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export default function sigmoid(x) {
  _sigmoid(x.tensor)
  return this
}

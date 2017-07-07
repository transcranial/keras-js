import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from './Tensor'

const _softsign = cwise({
  args: ['array'],
  body: function(_x) {
    _x /= 1 + Math.abs(_x)
  }
})

/**
 * Softsign activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export default function softsign(x) {
  _softsign(x.tensor)
  return this
}

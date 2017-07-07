import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from '../Tensor'

const _softplus = cwise({
  args: ['array'],
  body: function(_x) {
    _x = Math.log(Math.exp(_x) + 1)
  }
})

/**
 * Softplus activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export default function softplus(x) {
  _softplus(x.tensor)
  return this
}

import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from '../Tensor'

const _tanh = cwise({
  args: ['array'],
  body: function(_x) {
    _x = Math.tanh(_x)
  }
})

/**
 * Tanh activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export default function tanh(x) {
  _tanh(x.tensor)
  return this
}

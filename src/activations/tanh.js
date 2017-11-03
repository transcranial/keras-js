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
 * In-place operation: tanh activation function
 *
 * @param {Tensor} x
 */
export default function tanh(x) {
  _tanh(x.tensor)
}

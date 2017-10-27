import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from '../Tensor'

const _selu = cwise({
  args: ['array', 'scalar'],
  body: function(_x) {
    const alpha = 1.6732632423543772848170429916717
    const scale = 1.0507009873554804934193349852946
    _x = scale * (Math.max(_x, 0) + alpha * (Math.exp(Math.min(_x, 0)) - 1))
  }
})

/**
 * SELU activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export default function selu(x) {
  _selu(x.tensor)
  return this
}

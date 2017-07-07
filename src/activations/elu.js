import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from '../Tensor'

const _elu = cwise({
  args: ['array', 'scalar'],
  body: function(_x, alpha) {
    _x = Math.max(_x, 0) + alpha * (Math.exp(Math.min(_x, 0)) - 1)
  }
})

/**
 * ELU activation function. In-place operation.
 * @param {Tensor} x
 * @param {Number} opts.alpha
 * @returns {Tensor} `this`
 */
export default function elu(x, opts = {}) {
  const { alpha = 1.0 } = opts
  _elu(x.tensor, alpha)
  return this
}

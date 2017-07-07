import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from '../Tensor'

/**
 * ReLU activation function. In-place operation.
 * @param {Tensor} x
 * @param {Number} opts.alpha
 * @param {Number} opts.maxValue
 * @returns {Tensor} `this`
 */
export default function relu(x, opts = {}) {
  const { alpha = 0, maxValue = null } = opts
  let neg
  if (alpha !== 0) {
    neg = new Tensor([], x.tensor.shape)
    ops.mins(neg.tensor, x.tensor, 0)
    ops.mulseq(neg.tensor, alpha)
  }
  ops.maxseq(x.tensor, 0)
  if (maxValue) {
    ops.minseq(x.tensor, maxValue)
  }
  if (neg) {
    ops.addeq(x.tensor, neg.tensor)
  }
  return this
}

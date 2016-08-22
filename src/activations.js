import ndarray from 'ndarray'
import ops from 'ndarray-ops'

/**
 * Softmax activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export function softmax (x) {
  if (x.tensor.shape.length === 1) {
    ops.expeq(x.tensor)
    const sum = ops.sum(x.tensor)
    ops.divseq(x.tensor, sum)
  } else if (x.tensor.shape.length === 2) {
    for (let i = 0; i < x.tensor.shape[0]; i++) {
      ops.expeq(x.tensor.pick(i, null))
      const sum = ops.sum(x.tensor.pick(i, null))
      ops.divseq(x.tensor.pick(i, null), sum)
    }
  } else {
    throw new Error(`[activations.softmax] tensor shape ${x.tensor.shape} not supported.`)
  }
  return this
}

export function softplus (x) {

}

export function softsign (x) {

}

/**
 * ReLU activation function. In-place operation.
 * @param {Tensor} x
 * @param {Number} alpha
 * @param {Number} maxValue
 * @returns {Tensor} `this`
 */
export function relu (x, opts = {}) {
  const { alpha = 0, maxValue = null } = opts
  let neg
  if (alpha !== 0) {
    neg = ndarray(new x._type(x.tensor.data.length), x.tensor.shape)
    ops.mins(neg, x.tensor, 0.0)
    ops.mulseq(neg, alpha)
  }
  ops.maxseq(x.tensor, 0.0)
  if (maxValue) {
    ops.minseq(x.tensor, maxValue)
  }
  if (neg) {
    ops.addeq(x.tensor, neg)
  }
  return this
}

export function tanh (x) {

}

export function sigmoid (x) {

}

export function hardSigmoid (x) {

}

export function linear (x) {
  return x
}

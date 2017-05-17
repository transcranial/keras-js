import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from './Tensor'

/**
 * Softmax activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export function softmax(x) {
  if (x.tensor.shape.length === 1) {
    const maxval = ops.sup(x.tensor)
    ops.subseq(x.tensor, maxval)
    ops.expeq(x.tensor)
    const sum = ops.sum(x.tensor)
    ops.divseq(x.tensor, sum)
  } else if (x.tensor.shape.length === 2) {
    for (let i = 0; i < x.tensor.shape[0]; i++) {
      const maxval = ops.sup(x.tensor.pick(i, null))
      ops.subseq(x.tensor.pick(i, null), maxval)
      ops.expeq(x.tensor.pick(i, null))
      const sum = ops.sum(x.tensor.pick(i, null))
      ops.divseq(x.tensor.pick(i, null), sum)
    }
  } else {
    throw new Error(`[activations.softmax] tensor shape ${x.tensor.shape} not supported.`)
  }
  return this
}

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
export function elu(x, opts = {}) {
  const { alpha = 1.0 } = opts
  _elu(x.tensor, alpha)
  return this
}

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
export function softplus(x) {
  _softplus(x.tensor)
  return this
}

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
export function softsign(x) {
  _softsign(x.tensor)
  return this
}

/**
 * ReLU activation function. In-place operation.
 * @param {Tensor} x
 * @param {Number} opts.alpha
 * @param {Number} opts.maxValue
 * @returns {Tensor} `this`
 */
export function relu(x, opts = {}) {
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
export function tanh(x) {
  _tanh(x.tensor)
  return this
}

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
export function sigmoid(x) {
  _sigmoid(x.tensor)
  return this
}

// Reference hard sigmoid with slope and shift values from theano, see
// https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
const _hard_sigmoid = cwise({
  args: ['array'],
  body: function(_x) {
    _x = _x * 0.2 + 0.5
    if (_x <= 0) {
      _x = 0
    } else if (_x >= 1) {
      _x = 1
    }
  }
})

/**
 * Hard-sigmoid activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export function hard_sigmoid(x) {
  _hard_sigmoid(x.tensor)
  return this
}

/**
 * Linear activation function. In-place operation.
 * @param {Tensor} x
 * @returns {Tensor} `this`
 */
export function linear(x) {
  return this
}

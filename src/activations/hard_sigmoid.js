import ops from 'ndarray-ops'
import cwise from 'cwise'
import Tensor from '../Tensor'

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
export default function hard_sigmoid(x) {
  _hard_sigmoid(x.tensor)
  return this
}

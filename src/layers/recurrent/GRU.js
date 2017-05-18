import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'
import cwise from 'cwise'

/**
 * GRU layer class
 */
export default class GRU extends Layer {
  /**
   * Creates a GRU layer
   * @param {number} attrs.units - output dimensionality
   * @param {number} [attrs.activation] - activation function
   * @param {number} [attrs.recurrent_activation] - inner activation function
   * @param {number} [attrs.use_bias] - use bias
   * @param {number} [attrs.return_sequences] - return the last output in the output sequence or the full sequence
   * @param {number} [attrs.go_backwards] - process the input sequence backwards
   * @param {number} [attrs.stateful] - whether to save the last state as the initial state for the next pass
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'GRU'

    const {
      units = 1,
      activation = 'tanh',
      use_bias = true,
      recurrent_activation = 'hard_sigmoid',
      return_sequences = false,
      go_backwards = false,
      stateful = false
    } = attrs

    this.units = units

    // keep this.activation and this.recurrentActivation for Bidirectional wrapper layer to use
    this.activation = activation
    this.recurrentActivation = recurrent_activation
    this.activationFunc = activations[activation]
    this.recurrentActivationFunc = activations[recurrent_activation]

    this.use_bias = use_bias

    this.returnSequences = return_sequences
    this.goBackwards = go_backwards
    this.stateful = stateful

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'recurrent_kernel', 'bias'] : ['kernel', 'recurrent_kernel']
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   * W weight tensor is split into W_z, W_r, W_h
   * U weight tensor is split into U_z, U_r, U_h
   * b weight tensor is split into b_z, b_r, b_h (or create empty bias if this.use_bias is false)
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr)

    const shape_W = this.weights['kernel'].tensor.shape
    this.weights['W_z'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_r'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_h'] = new Tensor([], [shape_W[0], this.units])
    ops.assign(this.weights['W_z'].tensor, this.weights['kernel'].tensor.hi(shape_W[0], this.units).lo(0, 0))
    ops.assign(
      this.weights['W_r'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 2 * this.units).lo(0, this.units)
    )
    ops.assign(
      this.weights['W_h'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 3 * this.units).lo(0, 2 * this.units)
    )

    const shape_U = this.weights['recurrent_kernel'].tensor.shape
    this.weights['U_z'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_r'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_h'] = new Tensor([], [shape_U[0], this.units])
    ops.assign(this.weights['U_z'].tensor, this.weights['recurrent_kernel'].tensor.hi(shape_U[0], this.units).lo(0, 0))
    ops.assign(
      this.weights['U_r'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 2 * this.units).lo(0, this.units)
    )
    ops.assign(
      this.weights['U_h'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 3 * this.units).lo(0, 2 * this.units)
    )

    this.weights['b_z'] = new Tensor([], [this.units])
    this.weights['b_r'] = new Tensor([], [this.units])
    this.weights['b_h'] = new Tensor([], [this.units])
    if (this.use_bias) {
      ops.assign(this.weights['b_z'].tensor, this.weights['bias'].tensor.hi(this.units).lo(0))
      ops.assign(this.weights['b_r'].tensor, this.weights['bias'].tensor.hi(2 * this.units).lo(this.units))
      ops.assign(this.weights['b_h'].tensor, this.weights['bias'].tensor.hi(3 * this.units).lo(2 * this.units))
    }
  }

  _combine = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function(_y, _x1, _x2, _b) {
      _y = _x1 + _x2 + _b
    }
  })

  _update = cwise({
    args: ['array', 'array', 'array'],
    body: function(_h, _htm1, _z) {
      _h = _h * (1 - _z) + _htm1 * _z
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    let currentX = new Tensor([], [x.tensor.shape[1]])

    const dimUpdateGate = this.units
    const dimResetGate = this.units
    const dimHiddenState = this.units

    let currentUpdateGateState = new Tensor([], [dimUpdateGate])
    let tempXZ = new Tensor([], [dimUpdateGate])
    let tempHZ = new Tensor([], [dimUpdateGate])

    let currentResetGateState = new Tensor([], [dimResetGate])
    let tempXR = new Tensor([], [dimResetGate])
    let tempHR = new Tensor([], [dimResetGate])

    let currentHiddenState = this.stateful && this.currentHiddenState
      ? this.currentHiddenState
      : new Tensor([], [dimHiddenState])
    let tempXH = new Tensor([], [dimHiddenState])
    let tempHH = new Tensor([], [dimHiddenState])
    let previousHiddenState = new Tensor([], [dimHiddenState])

    this.hiddenStateSequence = new Tensor([], [x.tensor.shape[0], dimHiddenState])

    const _clearTemp = () => {
      const tempTensors = [tempXZ, tempHZ, tempXR, tempHR, tempXH, tempHH]
      tempTensors.forEach(temp => ops.assigns(temp.tensor, 0))
    }

    const _step = () => {
      ops.assign(previousHiddenState.tensor, currentHiddenState.tensor)

      gemv(1, this.weights['W_z'].tensor.transpose(1, 0), currentX.tensor, 1, tempXZ.tensor)
      gemv(1, this.weights['U_z'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHZ.tensor)
      this._combine(currentUpdateGateState.tensor, tempXZ.tensor, tempHZ.tensor, this.weights['b_z'].tensor)
      this.recurrentActivationFunc(currentUpdateGateState)

      gemv(1, this.weights['W_r'].tensor.transpose(1, 0), currentX.tensor, 1, tempXR.tensor)
      gemv(1, this.weights['U_r'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHR.tensor)
      this._combine(currentResetGateState.tensor, tempXR.tensor, tempHR.tensor, this.weights['b_r'].tensor)
      this.recurrentActivationFunc(currentResetGateState)

      ops.muleq(currentResetGateState.tensor, previousHiddenState.tensor)
      gemv(1, this.weights['W_h'].tensor.transpose(1, 0), currentX.tensor, 1, tempXH.tensor)
      gemv(1, this.weights['U_h'].tensor.transpose(1, 0), currentResetGateState.tensor, 1, tempHH.tensor)
      this._combine(currentHiddenState.tensor, tempXH.tensor, tempHH.tensor, this.weights['b_h'].tensor)
      this.activationFunc(currentHiddenState)

      this._update(currentHiddenState.tensor, previousHiddenState.tensor, currentUpdateGateState.tensor)
    }

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i
      ops.assign(currentX.tensor, x.tensor.pick(inputIndex, null))
      _clearTemp()
      _step()

      if (this.returnSequences) {
        ops.assign(this.hiddenStateSequence.tensor.pick(i, null), currentHiddenState.tensor)
      }
    }

    if (this.returnSequences) {
      x.tensor = this.hiddenStateSequence.tensor
    } else {
      x.tensor = currentHiddenState.tensor
    }

    if (this.stateful) {
      this.currentHiddenState = currentHiddenState
    }

    return x
  }
}

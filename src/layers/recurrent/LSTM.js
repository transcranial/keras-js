import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'
import cwise from 'cwise'

/**
 * LSTM layer class
 */
export default class LSTM extends Layer {
  /**
   * Creates a LSTM layer
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
    this.layerClass = 'LSTM'

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
   * W weight tensor is split into W_i, W_f, W_c, W_o
   * U weight tensor is split into U_i, U_f, U_c, U_o
   * b weight tensor is split into b_i, b_f, b_c, b_o (or create empty bias if this.use_bias is false)
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr)

    const shape_W = this.weights['kernel'].tensor.shape
    this.weights['W_i'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_f'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_c'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_o'] = new Tensor([], [shape_W[0], this.units])
    ops.assign(this.weights['W_i'].tensor, this.weights['kernel'].tensor.hi(shape_W[0], this.units).lo(0, 0))
    ops.assign(
      this.weights['W_f'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 2 * this.units).lo(0, this.units)
    )
    ops.assign(
      this.weights['W_c'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 3 * this.units).lo(0, 2 * this.units)
    )
    ops.assign(
      this.weights['W_o'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 4 * this.units).lo(0, 3 * this.units)
    )

    const shape_U = this.weights['recurrent_kernel'].tensor.shape
    this.weights['U_i'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_f'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_c'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_o'] = new Tensor([], [shape_U[0], this.units])
    ops.assign(this.weights['U_i'].tensor, this.weights['recurrent_kernel'].tensor.hi(shape_U[0], this.units).lo(0, 0))
    ops.assign(
      this.weights['U_f'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 2 * this.units).lo(0, this.units)
    )
    ops.assign(
      this.weights['U_c'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 3 * this.units).lo(0, 2 * this.units)
    )
    ops.assign(
      this.weights['U_o'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 4 * this.units).lo(0, 3 * this.units)
    )

    this.weights['b_i'] = new Tensor([], [this.units])
    this.weights['b_f'] = new Tensor([], [this.units])
    this.weights['b_c'] = new Tensor([], [this.units])
    this.weights['b_o'] = new Tensor([], [this.units])
    if (this.use_bias) {
      ops.assign(this.weights['b_i'].tensor, this.weights['bias'].tensor.hi(this.units).lo(0))
      ops.assign(this.weights['b_f'].tensor, this.weights['bias'].tensor.hi(2 * this.units).lo(this.units))
      ops.assign(this.weights['b_c'].tensor, this.weights['bias'].tensor.hi(3 * this.units).lo(2 * this.units))
      ops.assign(this.weights['b_o'].tensor, this.weights['bias'].tensor.hi(4 * this.units).lo(3 * this.units))
    }
  }

  _combine = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function(_y, _x1, _x2, _b) {
      _y = _x1 + _x2 + _b
    }
  })

  _update = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function(_c, _ctm1, _i, _f) {
      _c = _c * _i + _ctm1 * _f
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    let currentX = new Tensor([], [x.tensor.shape[1]])

    const dimInputGate = this.weights['b_i'].tensor.shape[0]
    const dimCandidate = this.weights['b_c'].tensor.shape[0]
    const dimForgetGate = this.weights['b_f'].tensor.shape[0]
    const dimOutputGate = this.weights['b_o'].tensor.shape[0]

    let currentInputGateState = new Tensor([], [dimInputGate])
    let tempXI = new Tensor([], [dimInputGate])
    let tempHI = new Tensor([], [dimInputGate])

    let currentForgetGateState = new Tensor([], [dimForgetGate])
    let tempXF = new Tensor([], [dimForgetGate])
    let tempHF = new Tensor([], [dimForgetGate])

    let currentOutputGateState = new Tensor([], [dimOutputGate])
    let tempXO = new Tensor([], [dimOutputGate])
    let tempHO = new Tensor([], [dimOutputGate])

    let currentCandidate = new Tensor([], [dimCandidate])
    let tempXC = new Tensor([], [dimCandidate])
    let tempHC = new Tensor([], [dimCandidate])
    let previousCandidate = this.stateful && this.previousCandidate
      ? this.previousCandidate
      : new Tensor([], [dimCandidate])

    let currentHiddenState = this.stateful && this.currentHiddenState
      ? this.currentHiddenState
      : new Tensor([], [dimCandidate])
    let previousHiddenState = new Tensor([], [dimCandidate])

    this.hiddenStateSequence = new Tensor([], [x.tensor.shape[0], dimCandidate])

    const _clearTemp = () => {
      const tempTensors = [tempXI, tempHI, tempXF, tempHF, tempXO, tempHO, tempXC, tempHC]
      tempTensors.forEach(temp => ops.assigns(temp.tensor, 0))
    }

    const _step = () => {
      ops.assign(previousHiddenState.tensor, currentHiddenState.tensor)

      gemv(1, this.weights['W_i'].tensor.transpose(1, 0), currentX.tensor, 1, tempXI.tensor)
      gemv(1, this.weights['U_i'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHI.tensor)
      this._combine(currentInputGateState.tensor, tempXI.tensor, tempHI.tensor, this.weights['b_i'].tensor)
      this.recurrentActivationFunc(currentInputGateState)

      gemv(1, this.weights['W_f'].tensor.transpose(1, 0), currentX.tensor, 1, tempXF.tensor)
      gemv(1, this.weights['U_f'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHF.tensor)
      this._combine(currentForgetGateState.tensor, tempXF.tensor, tempHF.tensor, this.weights['b_f'].tensor)
      this.recurrentActivationFunc(currentForgetGateState)

      gemv(1, this.weights['W_o'].tensor.transpose(1, 0), currentX.tensor, 1, tempXO.tensor)
      gemv(1, this.weights['U_o'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHO.tensor)
      this._combine(currentOutputGateState.tensor, tempXO.tensor, tempHO.tensor, this.weights['b_o'].tensor)
      this.recurrentActivationFunc(currentOutputGateState)

      gemv(1, this.weights['W_c'].tensor.transpose(1, 0), currentX.tensor, 1, tempXC.tensor)
      gemv(1, this.weights['U_c'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHC.tensor)
      this._combine(currentCandidate.tensor, tempXC.tensor, tempHC.tensor, this.weights['b_c'].tensor)
      this.activationFunc(currentCandidate)

      this._update(
        currentCandidate.tensor,
        previousCandidate.tensor,
        currentInputGateState.tensor,
        currentForgetGateState.tensor
      )

      ops.assign(previousCandidate.tensor, currentCandidate.tensor)

      this.activationFunc(currentCandidate)
      ops.mul(currentHiddenState.tensor, currentOutputGateState.tensor, currentCandidate.tensor)
    }

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i
      ops.assign(currentX.tensor, x.tensor.pick(inputIndex, null))
      _clearTemp()
      _step()

      ops.assign(this.hiddenStateSequence.tensor.pick(i, null), currentHiddenState.tensor)
    }

    if (this.returnSequences) {
      x.tensor = this.hiddenStateSequence.tensor
    } else {
      x.tensor = currentHiddenState.tensor
    }

    if (this.stateful) {
      this.previousCandidate = previousCandidate
      this.currentHiddenState = currentHiddenState
    }

    return x
  }
}

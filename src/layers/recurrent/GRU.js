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
   * @param {number} attrs.outputDim - output dimensionality
   * @param {number} [attrs.activation] - activation function
   * @param {number} [attrs.innerActivation] - inner activation function
   * @param {number} [attrs.returnSequences] - return the last output in the output sequence or the full sequence
   * @param {number} [attrs.goBackwards] - process the input sequence backwards
   * @param {number} [attrs.stateful] - whether to save the last state as the initial state for the next pass
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'GRU'

    const {
      outputDim = 1,
      activation = 'tanh',
      innerActivation = 'hard_sigmoid',
      returnSequences = false,
      goBackwards = false,
      stateful = false
    } = attrs

    this.outputDim = outputDim

    // keep this.activation and this.innerActivation for Bidirectional wrapper layer to use
    this.activation = activation
    this.innerActivation = innerActivation
    this.activationFunc = activations[activation]
    this.innerActivationFunc = activations[innerActivation]

    this.returnSequences = returnSequences
    this.goBackwards = goBackwards
    this.stateful = stateful

    // Layer weights specification
    this.params = ['W_z', 'U_z', 'b_z', 'W_r', 'U_r', 'b_r', 'W_h', 'U_h', 'b_h']
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

    const dimUpdateGate = this.weights['b_z'].tensor.shape[0]
    const dimResetGate = this.weights['b_r'].tensor.shape[0]
    const dimHiddenState = this.weights['b_h'].tensor.shape[0]

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
      this.innerActivationFunc(currentUpdateGateState)

      gemv(1, this.weights['W_r'].tensor.transpose(1, 0), currentX.tensor, 1, tempXR.tensor)
      gemv(1, this.weights['U_r'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHR.tensor)
      this._combine(currentResetGateState.tensor, tempXR.tensor, tempHR.tensor, this.weights['b_r'].tensor)
      this.innerActivationFunc(currentResetGateState)

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

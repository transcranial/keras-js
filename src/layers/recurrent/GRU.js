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
   * @param {Object} [attrs] - layer attributes
   */
  constructor (attrs = {}) {
    super(attrs)
    this.layerClass = 'GRU'

    const {
      outputDim = 1,
      activation = 'tanh',
      innerActivation = 'hardSigmoid'
    } = attrs

    this.outputDim = outputDim

    this.activation = activations[activation]
    this.innerActivation = activations[innerActivation]

    // Layer weights specification
    this.params = ['W_z', 'U_z', 'b_z', 'W_r', 'U_r', 'b_r', 'W_h', 'U_h', 'b_h']
  }

  _combine = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function (_y, _x1, _x2, _b) {
      _y = _x1 + _x2 + _b
    }
  })

  _update = cwise({
    args: ['array', 'array', 'array'],
    body: function (_h, _htm1, _z) {
      _h = _h * (1 - _z) + _htm1 * _z
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
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

    let currentHiddenState = new Tensor([], [dimHiddenState])
    let tempXH = new Tensor([], [dimHiddenState])
    let tempHH = new Tensor([], [dimHiddenState])
    let previousHiddenState = new Tensor([], [dimHiddenState])

    const _clearTemp = () => {
      const tempTensors = [tempXZ, tempHZ, tempXR, tempHR, tempXH, tempHH]
      tempTensors.forEach(temp => ops.assigns(temp.tensor, 0))
    }

    const _step = () => {
      ops.assign(previousHiddenState.tensor, currentHiddenState.tensor)

      gemv(1.0, this.weights['W_z'].tensor.transpose(1, 0), currentX.tensor, 1.0, tempXZ.tensor)
      gemv(1.0, this.weights['U_z'].tensor.transpose(1, 0), previousHiddenState.tensor, 1.0, tempHZ.tensor)
      this._combine(currentUpdateGateState.tensor, tempXZ.tensor, tempHZ.tensor, this.weights['b_z'].tensor)
      this.innerActivation(currentUpdateGateState)

      gemv(1.0, this.weights['W_r'].tensor.transpose(1, 0), currentX.tensor, 1.0, tempXR.tensor)
      gemv(1.0, this.weights['U_r'].tensor.transpose(1, 0), previousHiddenState.tensor, 1.0, tempHR.tensor)
      this._combine(currentResetGateState.tensor, tempXR.tensor, tempHR.tensor, this.weights['b_r'].tensor)
      this.innerActivation(currentResetGateState)

      ops.muleq(currentResetGateState.tensor, previousHiddenState.tensor)
      gemv(1.0, this.weights['W_h'].tensor.transpose(1, 0), currentX.tensor, 1.0, tempXH.tensor)
      gemv(1.0, this.weights['U_h'].tensor.transpose(1, 0), currentResetGateState.tensor, 1.0, tempHH.tensor)
      this._combine(currentHiddenState.tensor, tempXH.tensor, tempHH.tensor, this.weights['b_h'].tensor)
      this.activation(currentHiddenState)

      this._update(
        currentHiddenState.tensor,
        previousHiddenState.tensor,
        currentUpdateGateState.tensor
      )
    }

    for (let i = 0, steps = x.tensor.shape[0]; i < steps; i++) {
      ops.assign(currentX.tensor, x.tensor.pick(i, null))
      _clearTemp()
      _step()
    }

    x.tensor = currentHiddenState.tensor

    return x
  }
}

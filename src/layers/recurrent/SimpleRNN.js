import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'
import cwise from 'cwise'

/**
 * SimpleRNN layer class
 */
export default class SimpleRNN extends Layer {
  /**
   * Creates a SimpleRNN layer
   * @param {number} attrs.outputDim - output dimensionality
   * @param {number} [attrs.activation] - activation function
   * @param {Object} [attrs] - layer attributes
   */
  constructor (attrs = {}) {
    super(attrs)
    this.layerClass = 'SimpleRNN'

    const {
      outputDim = 1,
      activation = 'tanh'
    } = attrs

    this.outputDim = outputDim

    this.activation = activations[activation]

    // Layer weights specification
    this.params = ['W', 'U', 'b']
  }

  _combine = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function (_y, _x1, _x2, _b) {
      _y = _x1 + _x2 + _b
    }
  })

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    let currentX = new Tensor([], [x.tensor.shape[1]])

    const dimHiddenState = this.weights['b'].tensor.shape[0]
    let currentHiddenState = new Tensor([], [dimHiddenState])
    let tempXH = new Tensor([], [dimHiddenState])
    let tempHH = new Tensor([], [dimHiddenState])
    let previousHiddenState = new Tensor([], [dimHiddenState])

    const _clearTemp = () => {
      const tempTensors = [tempXH, tempHH]
      tempTensors.forEach(temp => ops.assigns(temp.tensor, 0))
    }

    const _step = () => {
      ops.assign(previousHiddenState.tensor, currentHiddenState.tensor)

      gemv(1.0, this.weights['W'].tensor.transpose(1, 0), currentX.tensor, 1.0, tempXH.tensor)
      gemv(1.0, this.weights['U'].tensor.transpose(1, 0), previousHiddenState.tensor, 1.0, tempHH.tensor)
      this._combine(currentHiddenState.tensor, tempXH.tensor, tempHH.tensor, this.weights['b'].tensor)
      this.activation(currentHiddenState)
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

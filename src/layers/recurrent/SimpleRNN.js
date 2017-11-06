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
   *
   * @param {Object} [attrs] - layer attributes
   * @param {number} [attrs.units] - output dimensionality
   * @param {number} [attrs.activation] - activation function
   * @param {number} [attrs.use_bias] - use bias
   * @param {number} [attrs.return_sequences] - return the last output in the output sequence or the full sequence
   * @param {number} [attrs.go_backwards] - process the input sequence backwards
   * @param {number} [attrs.stateful] - whether to save the last state as the initial state for the next pass
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'SimpleRNN'

    const {
      units = 1,
      activation = 'tanh',
      use_bias = true,
      return_sequences = false,
      go_backwards = false,
      stateful = false
    } = attrs

    this.units = units

    // keep this.activation for Bidirectional wrapper layer to use
    this.activation = activation
    this.activationFunc = activations[activation]

    this.use_bias = use_bias

    this.returnSequences = return_sequences
    this.goBackwards = go_backwards
    this.stateful = stateful

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'recurrent_kernel', 'bias'] : ['kernel', 'recurrent_kernel']
  }

  /**
   * Method for setting layer weights. Extends `super` method. Create empty bias if this.use_bias is false.
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr)
    if (!this.use_bias) {
      this.weights['bias'] = new Tensor([], [this.units])
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.gpu) {
      this._callGPU(x)
    } else {
      this._callCPU(x)
    }
    return this.output
  }

  _combine = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function(_y, _x1, _x2, _b) {
      _y = _x1 + _x2 + _b
    }
  })

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    const dimHiddenState = this.units
    const currentHiddenState =
      this.stateful && this.currentHiddenState ? this.currentHiddenState : new Tensor([], [dimHiddenState])
    const tempXH = new Tensor([], [dimHiddenState])
    const tempHH = new Tensor([], [dimHiddenState])
    const previousHiddenState = new Tensor([], [dimHiddenState])

    this.hiddenStateSequence = new Tensor([], [x.tensor.shape[0], dimHiddenState])

    const current = new Tensor([], [x.tensor.shape[1]])

    const _step = () => {
      ops.assign(previousHiddenState.tensor, currentHiddenState.tensor)

      gemv(1, this.weights['kernel'].tensor.transpose(1, 0), current.tensor, 1, tempXH.tensor)
      gemv(1, this.weights['recurrent_kernel'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHH.tensor)
      this._combine(currentHiddenState.tensor, tempXH.tensor, tempHH.tensor, this.weights['bias'].tensor)
      this.activationFunc(currentHiddenState)
    }

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i
      ops.assign(current.tensor, x.tensor.pick(inputIndex, null))

      // clear temp tensors
      const tempTensors = [tempXH, tempHH]
      tempTensors.forEach(temp => ops.assigns(temp.tensor, 0))

      // advance timestep
      _step()

      if (this.returnSequences) {
        ops.assign(this.hiddenStateSequence.tensor.pick(i, null), currentHiddenState.tensor)
      }
    }

    if (this.returnSequences) {
      this.output = this.hiddenStateSequence
    } else {
      this.output = currentHiddenState
    }

    if (this.stateful) {
      this.currentHiddenState = currentHiddenState
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {}
}

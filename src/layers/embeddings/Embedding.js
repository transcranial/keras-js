import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * Embedding layer class
 */
export default class Embedding extends Layer {
  /**
   * Creates a Embedding layer
   */
  constructor (inputDim, outputDim, attrs = {}) {
    super(attrs)
    const {
      inputLength = 0,
      maskZero = false,
      dropout = 0.0
    } = attrs

    this.inputDim = inputDim
    this.outputDim = outputDim
    this.inputLength = inputLength

    // maskZero will be important for subsequence layers
    this.maskZero = maskZero

    // relevant only during training phase
    this.dropout = dropout

    // Layer weights specification
    this.params = ['W']
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    let y = new Tensor([], [x.tensor.shape[0], this.weights.W.tensor.shape[1]])

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      ops.assign(y.tensor.pick(i, null), this.weights.W.tensor.pick(x.tensor.get(i), null))
    }

    x.tensor = y.tensor
    return x
  }
}

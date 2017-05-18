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
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Embedding'

    const { input_dim = 1, output_dim = 1, input_length = 0, mask_zero = false } = attrs

    this.inputDim = input_dim
    this.outputDim = output_dim
    this.inputLength = input_length

    // mask_zero will be important for subsequent layers
    this.maskZero = mask_zero

    // Layer weights specification
    this.params = ['embeddings']
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    let y = new Tensor([], [x.tensor.shape[0], this.weights['embeddings'].tensor.shape[1]])

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      ops.assign(y.tensor.pick(i, null), this.weights['embeddings'].tensor.pick(x.tensor.get(i), null))
    }

    x.tensor = y.tensor
    return x
  }
}

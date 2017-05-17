import Layer from '../../Layer'
import unsqueeze from 'ndarray-unsqueeze'
import tile from 'ndarray-tile'

/**
 * RepeatVector layer class
 * Turns 2D tensors of shape [features] to 3D tensors of shape [n, features].
 * Note there is no concept of batch size in these layers (single-batch) so we're actually going from 1D to 2D.
 */
export default class RepeatVector extends Layer {
  /**
   * Creates a RepeatVector layer
   * @param {number} attrs.n
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'RepeatVector'

    const { n = 1 } = attrs
    this.n = n
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    if (x.tensor.shape.length !== 1) {
      throw new Error(`${this.name} [RepeatVector layer] Only 1D tensor inputs allowed.`)
    }
    x.tensor = tile(unsqueeze(x.tensor, 0), [this.n, 1])
    return x
  }
}

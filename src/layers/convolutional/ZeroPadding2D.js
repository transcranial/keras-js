import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * ZeroPadding2D layer class
 */
export default class ZeroPadding2D extends Layer {
  /**
   * Creates a ZeroPadding2D activation layer
   * @param {number} attrs.padding - size of padding
   */
  constructor (attrs = {}) {
    super(attrs)
    const {
      padding = [1, 1],
      dimOrdering = 'tf'
    } = attrs

    this.padding = padding
    this.dimOrdering = dimOrdering
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    // convert to tf ordering
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

    const inputShape = x.tensor.shape
    const outputShape = [
      inputShape[0] + this.padding[0] * 2,
      inputShape[1] + this.padding[1] * 2,
      inputShape[2]
    ]
    let y = new Tensor([], outputShape)
    ops.assign(
      y.tensor
        .hi(inputShape[0] + this.padding[0], inputShape[1] + this.padding[1], inputShape[2])
        .lo(this.padding[0], this.padding[1], 0),
      x.tensor
    )
    x.tensor = y.tensor

    // convert back to th ordering if necessary
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(2, 0, 1)
    }

    return x
  }
}

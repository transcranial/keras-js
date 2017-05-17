import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * ZeroPadding3D layer class
 */
export default class ZeroPadding3D extends Layer {
  /**
   * Creates a ZeroPadding3D layer
   * @param {Number|Array<Number>|Array<Array<Number>>} attrs.padding - int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints
   * @param {String} attrs.data_format - either 'channels_last' or 'channels_first'
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'ZeroPadding3D'

    const { padding = [[1, 1], [1, 1], [1, 1]], data_format = 'channels_last' } = attrs

    if (Array.isArray(padding)) {
      if (Array.isArray(padding[0])) {
        // [[int, int], [int, int], [int, int]]
        this.padding = padding
      } else {
        // [int, int, int]
        this.padding = [[padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]]]
      }
    } else {
      // int
      this.padding = [[padding, padding], [padding, padding], [padding, padding]]
    }

    this.dataFormat = data_format
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 3, 0)
    }

    const inputShape = x.tensor.shape
    const outputShape = [
      inputShape[0] + this.padding[0][0] + this.padding[0][1],
      inputShape[1] + this.padding[1][0] + this.padding[1][1],
      inputShape[2] + this.padding[2][0] + this.padding[2][1],
      inputShape[3]
    ]
    let y = new Tensor([], outputShape)
    ops.assign(
      y.tensor
        .hi(
          inputShape[0] + this.padding[0][0],
          inputShape[1] + this.padding[1][0],
          inputShape[2] + this.padding[2][0],
          inputShape[3]
        )
        .lo(this.padding[0][0], this.padding[1][0], this.padding[2][0], 0),
      x.tensor
    )
    x.tensor = y.tensor

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(3, 0, 1, 2)
    }

    return x
  }
}

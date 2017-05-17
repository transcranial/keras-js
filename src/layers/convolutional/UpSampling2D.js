import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * UpSampling2D layer class
 */
export default class UpSampling2D extends Layer {
  /**
   * Creates a UpSampling2D layer
   * @param {Number|Array<Number>} attrs.size - upsampling factor, int or tuple of int (length 2)
   * @param {String} attrs.data_format - either 'channels_last' or 'channels_first'
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'UpSampling2D'

    const { size = [2, 2], data_format = 'channels_last' } = attrs

    if (Array.isArray(size)) {
      this.size = size
    } else {
      this.size = [size, size]
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
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

    const inputShape = x.tensor.shape
    const outputShape = [inputShape[0] * this.size[0], inputShape[1] * this.size[1], inputShape[2]]
    let y = new Tensor([], outputShape)
    for (let i = 0; i < this.size[0]; i++) {
      for (let j = 0; j < this.size[1]; j++) {
        ops.assign(y.tensor.lo(i, j, 0).step(this.size[0], this.size[1], 1), x.tensor)
      }
    }
    x.tensor = y.tensor

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(2, 0, 1)
    }

    return x
  }
}

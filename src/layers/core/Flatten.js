import Layer from '../../Layer'
import ndarray from 'ndarray'
import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'

/**
 * Flatten layer class
 * Turns tensor into 1-d. Note there is no concept of batch size in these layers (single-batch).
 * We use ndarray-unpack first, as ndarray striding/offsets precludes us from simply using x.tensor.data
 */
export default class Flatten extends Layer {
  /**
   * Creates a Flatten layer
   */
  constructor (attrs = {}) {
    super(attrs)
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    if (x.tensor.shape.length > 1) {
      const shape = [x.tensor.shape.reduce((a, b) => a * b, 1)]
      x.tensor = ndarray(new x._type(flattenDeep(unpack(x.tensor))), shape)
    }
    return x
  }
}

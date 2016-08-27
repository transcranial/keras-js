import Layer from '../../engine/Layer'
import ndarray from 'ndarray'
import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'

/**
 * Reshape layer class
 * Note there is no concept of batch size in these layers (single-batch).
 * We use ndarray-unpack first, as ndarray striding/offsets precludes us from simply using x.tensor.data
 */
export default class Reshape extends Layer {
  /**
   * Creates a Reshape layer
   * @param {number[]} shape
   */
  constructor (shape, attrs = {}) {
    super(attrs)
    this.shape = shape
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call = x => {
    if (this.shape.reduce((a, b) => a * b, 1) !== x.tensor.size) {
      throw new Error(`${this.name} [Reshape layer] The total size of new array must be unchanged in reshape layer.`)
    }
    x.tensor = ndarray(new x._type(flattenDeep(unpack(x.tensor))), this.shape)
    return x
  }
}

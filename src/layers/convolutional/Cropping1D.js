import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * Cropping1D layer class
 */
export default class Cropping1D extends Layer {
  /**
   * Creates a Cropping1D layer
   * @param {Number|Array<Number>} attrs.cropping - int or tuple of int (length 2)
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Cropping1D'

    const { cropping = [0, 0] } = attrs

    if (Array.isArray(cropping)) {
      this.cropping = cropping
    } else {
      this.cropping = [cropping, cropping]
    }
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    const inputShape = x.tensor.shape
    const outputShape = [inputShape[0] - this.cropping[0] - this.cropping[1], inputShape[1]]
    let y = new Tensor([], outputShape)
    ops.assign(y.tensor, x.tensor.hi(inputShape[0] - this.cropping[1], inputShape[2]).lo(this.cropping[0], 0))
    x.tensor = y.tensor
    return x
  }
}

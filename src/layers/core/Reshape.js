import Tensor from '../../Tensor'
import Layer from '../../Layer'

/**
 * Reshape layer class
 * Note there is no concept of batch size in these layers (single-batch).
 */
export default class Reshape extends Layer {
  /**
   * Creates a Reshape layer
   * @param {number[]} attrs.target_shape
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Reshape'

    const { target_shape = [] } = attrs
    this.targetShape = target_shape
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    if (this.targetShape.reduce((a, b) => a * b, 1) !== x.tensor.size) {
      throw new Error(`${this.name} [Reshape layer] The total size of new array must be unchanged in reshape layer.`)
    }
    let reshaped = new Tensor([], this.targetShape)
    reshaped.replaceTensorData(x.tensor.data)
    x.tensor = reshaped.tensor
    return x
  }
}

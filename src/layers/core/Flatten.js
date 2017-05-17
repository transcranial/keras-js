import Tensor from '../../Tensor'
import Layer from '../../Layer'

/**
 * Flatten layer class
 * Turns tensor into 1-d. Note there is no concept of batch size in these layers (single-batch).
 */
export default class Flatten extends Layer {
  /**
   * Creates a Flatten layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Flatten'
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    if (x.tensor.shape.length > 1) {
      let raveled = new Tensor([], [x.tensor.shape.reduce((a, b) => a * b, 1)])
      raveled.replaceTensorData(x.tensor.data)
      x.tensor = raveled.tensor
    }
    return x
  }
}

import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * GlobalAveragePooling1D layer class
 */
export default class GlobalAveragePooling1D extends Layer {
  /**
   * Creates a GlobalAveragePooling1D layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'GlobalAveragePooling1D'
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    const [steps, features] = x.tensor.shape
    let y = new Tensor([], [features])
    for (let i = 0, len = features; i < len; i++) {
      y.tensor.set(i, ops.sum(x.tensor.pick(null, i)) / steps)
    }
    x.tensor = y.tensor
    return x
  }
}

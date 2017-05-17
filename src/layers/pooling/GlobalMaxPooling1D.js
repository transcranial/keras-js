import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * GlobalMaxPooling1D layer class
 */
export default class GlobalMaxPooling1D extends Layer {
  /**
   * Creates a GlobalMaxPooling1D layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'GlobalMaxPooling1D'
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    const features = x.tensor.shape[1]
    let y = new Tensor([], [features])
    for (let i = 0, len = features; i < len; i++) {
      y.tensor.set(i, ops.sup(x.tensor.pick(null, i)))
    }
    x.tensor = y.tensor
    return x
  }
}

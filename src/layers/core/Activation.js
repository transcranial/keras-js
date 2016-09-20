import * as activations from '../../activations'
import Layer from '../../Layer'

/**
 * Activation layer class
 */
export default class Activation extends Layer {
  /**
   * Creates an Activation layer
   * @param {string} attrs.activation - name of activation function
   */
  constructor (attrs = {}) {
    super(attrs)
    this.layerClass = 'Activation'

    const {
      activation = 'linear'
    } = attrs

    this.activation = activations[activation]
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    this.activation(x)
    return x
  }
}

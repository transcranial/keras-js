import Layer from './Layer'
import isEqual from 'lodash/isEqual'

/**
 * Input layer class
 */
export default class Input extends Layer {
  /**
   * Creates an Input layer
   */
  constructor (attrs = {}) {
    super(attrs)

    const {
      inputShape = []
    } = attrs

    this.inputShape = inputShape
  }

  call (x) {
    if (!isEqual(x.tensor.shape, this.inputShape)) {
      throw new Error(`[Input] input tensor shape does not match specified shape.`)
    }
    return x
  }
}

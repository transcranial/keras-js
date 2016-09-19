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
      shape = []
    } = attrs

    this.shape = shape
  }

  call (x) {
    if (!isEqual(x.tensor.shape, this.shape)) {
      throw new Error(`[Input] input tensor shape ${x.tensor.shape} does not match specified shape ${this.shape}.`)
    }
    return x
  }
}

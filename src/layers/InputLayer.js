import Layer from '../Layer'
import isEqual from 'lodash/isEqual'

/**
 * InputLayer layer class
 */
export default class InputLayer extends Layer {
  /**
   * Creates an InputLayer layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'InputLayer'

    const { shape = [] } = attrs

    this.shape = attrs.batch_input_shape && attrs.batch_input_shape.length ? attrs.batch_input_shape.slice(1) : shape
  }

  call(x) {
    if (!isEqual(x.tensor.shape, this.shape)) {
      throw new Error(`[InputLayer] input tensor shape ${x.tensor.shape} does not match specified shape ${this.shape}.`)
    }
    return x
  }
}

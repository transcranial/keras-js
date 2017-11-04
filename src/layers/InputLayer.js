import Layer from '../Layer'
import isEqual from 'lodash/isEqual'

/**
 * InputLayer layer class
 */
export default class InputLayer extends Layer {
  /**
   * Creates an InputLayer layer
   *
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'InputLayer'

    const { shape = [] } = attrs

    this.shape = attrs.batch_input_shape && attrs.batch_input_shape.length ? attrs.batch_input_shape.slice(1) : shape
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.gpu) {
      this._callGPU(x)
    } else {
      this._callCPU(x)
    }
    return this.output
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    this.inputShape = x.tensor.shape
    if (!isEqual(this.inputShape, this.shape)) {
      throw new Error(`[InputLayer] input tensor shape ${x.tensor.shape} does not match specified shape ${this.shape}.`)
    }
    this.output = x
  }

  /**
 * GPU call
 *
 * @param {Tensor} x
 */
  _callGPU(x) {
    if (!x.glTexture) {
      this.inputShape = x.tensor.shape
    } else {
      this.inputShape = x.originalShape
    }

    if (!isEqual(this.inputShape, this.shape)) {
      throw new Error(`[InputLayer] input tensor shape ${x.tensor.shape} does not match specified shape ${this.shape}.`)
    }

    if (!x.glTexture) {
      if (x.tensor.shape.length <= 2) {
        x.createGLTexture()
      } else if (x.tensor.shape.length > 2) {
        x.reshapeTo2D()
        x.createGLTexture()
      }
    }

    this.output = x
  }
}

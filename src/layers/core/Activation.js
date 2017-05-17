import * as activations from '../../activations'
import Layer from '../../Layer'
import checkPipelineSupport from '../../utils/checkPipelineSupport'
import WebGLActivation from '../../ext/core/WebGLActivation'

/**
 * Activation layer class
 */
export default class Activation extends Layer {
  /**
   * Creates an Activation layer
   * @param {string} attrs.activation - name of activation function
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Activation'

    const { activation = 'linear' } = attrs

    this.activation = activation
    this.activationFunc = activations[activation]

    // Enable layer gpu +/- pipeline mode if supported
    if (this.gpu && weblas) {
      this._useWeblas = true
      if (this.pipeline) {
        const isPipelineModeSupported = checkPipelineSupport(this.layerClass, attrs)
        if (isPipelineModeSupported) {
          this._pipelineEnabled = true
          this.webglActivation = new WebGLActivation()
        } else {
          this._pipelineEnabled = false
        }
      }
    }
  }

  /**
   * Runs layer computational logic in pipeline mode
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callPipelineMode(x) {
    if (!x._fromPipeline) {
      return this._callRegularMode(x)
    }
    x.weblasTensor = this.webglActivation.call(x.weblasTensor, this.activation)
    return x
  }

  /**
   * Runs layer computational logic in regular mode
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callRegularMode(x) {
    this.activationFunc(x)
    return x
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    if (this._pipelineEnabled) {
      return this._callPipelineMode(x)
    } else {
      return this._callRegularMode(x)
    }
  }
}

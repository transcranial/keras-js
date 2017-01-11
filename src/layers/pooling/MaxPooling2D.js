import _Pooling2D from './_Pooling2D'
import checkPipelineSupport from '../../utils/checkPipelineSupport'
import WebGLPooling2D from '../../ext/pooling/WebGLPooling2D'

/**
 * MaxPooling2D layer class, extends abstract _Pooling2D class
 */
export default class MaxPooling2D extends _Pooling2D {
  /**
   * Creates a MaxPooling2D layer
   */
  constructor (attrs = {}) {
    super(attrs)
    this.layerClass = 'MaxPooling2D'

    this.poolingFunc = 'max'

    // Enable layer pipeline mode if supported
    if (this._useWeblas) {
      const isPipelineModeSupported = checkPipelineSupport(this.layerClass, attrs)
      if (isPipelineModeSupported) {
        this._pipelineEnabled = true
        this.webglPooling2D = new WebGLPooling2D(this.poolingFunc)
      }
    }
  }
}

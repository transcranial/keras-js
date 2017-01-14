import _Pooling2D from './_Pooling2D';
import checkPipelineSupport from '../../utils/checkPipelineSupport';
import WebGLPooling2D from '../../ext/pooling/WebGLPooling2D';

/**
 * AveragePooling2D layer class, extends abstract _Pooling2D class
 */
export default class AveragePooling2D extends _Pooling2D {
  /**
   * Creates a AveragePooling2D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'AveragePooling2D';

    this.poolingFunc = 'average';

    // Enable layer pipeline mode if supported
    if (this._useWeblas && this._pipelineEnabled) {
      const isPipelineModeSupported = checkPipelineSupport(
        this.layerClass,
        attrs
      );
      if (!isPipelineModeSupported) {
        this._pipelineEnabled = false;
      } else {
        this.webglPooling2D = new WebGLPooling2D(this.poolingFunc);
      }
    }
  }
}

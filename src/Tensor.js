import ndarray from 'ndarray';
import squeeze from 'ndarray-squeeze';
import { MAX_TEXTURE_SIZE } from './constants';

const checkShape = (data, shape) => {
  if (
    data.length && shape.length &&
      data.length !== shape.reduce((a, b) => a * b, 1)
  ) {
    throw new Error('Specified shape incompatible with data.');
  }
};

/**
 * Tensor class
 */
export default class Tensor {
  /**
   * Creates a tensor
   * @param {(TypedArray|Array)} data
   * @param {Array} shape
   * @param {Object} [options]
   */
  constructor(data, shape, options = {}) {
    this._type = options.type || Float32Array;

    if (
      data && data.length &&
        (data instanceof this._type || data instanceof Array)
    ) {
      checkShape(data, shape);
      this.tensor = ndarray(data, shape);
      this.tensor = ndarray(new this._type(data), shape);
    } else if (!data.length && shape.length) {
      // if shape present but data not provided, initialize with 0s
      this.tensor = ndarray(
        new this._type(shape.reduce((a, b) => a * b, 1)),
        shape
      );
    } else {
      this.tensor = ndarray(new this._type([]), []);
    }

    this.webgl = weblas.gpu.gl;

    this._gpuMaxSizeExceeded = false;
  }

  /**
   * Replaces data in the underlying ndarray.
   */
  replaceTensorData(data) {
    if (data && data.length && data instanceof this._type) {
      this.tensor.data = data;
    } else if (data && data.length && data instanceof Array) {
      this.tensor.data = new this._type(data);
    } else {
      throw new Error('[Tensor] invalid input for replaceTensorData method.');
    }
  }

  /**
   * Create weblas pipeline tensor in GPU memory
   * 1-D or 2-D only
   * see https://github.com/waylonflinn/weblas/wiki/Pipeline
   *
   * gl.MAX_TEXTURE_SIZE is a limiting factor.
   * Where this is exceeded, falls back to CPU.
   */
  createWeblasTensor() {
    if (this.weblasTensor) {
      this.weblasTensor.delete();
    }

    if (this.tensor.shape.length === 1) {
      const len = this.tensor.shape[0];
      if (len > MAX_TEXTURE_SIZE) {
        this._gpuMaxSizeExceeded = true;
      } else {
        const shape = [ 1, len ];
        this.weblasTensor = new weblas.pipeline.Tensor(shape, this.tensor.data);
      }
    } else if (this.tensor.shape.length === 2) {
      if (this.tensor.shape.some(s => s > MAX_TEXTURE_SIZE)) {
        this._gpuMaxSizeExceeded = true;
      } else {
        const shape = this.tensor.shape;
        this.weblasTensor = new weblas.pipeline.Tensor(shape, this.tensor.data);
      }
    } else {
      throw new Error(
        '[Tensor] can only create weblas Tensor for 1-D or 2-D only'
      );
    }
  }

  /**
   * Copies from another weblas pipeline tensor
   */
  copyFromWeblasTensor(weblasTensor) {
    const gl = this.webgl.context;
    this.weblasTensor = new weblas.pipeline.Tensor(weblasTensor.shape, null);

    gl.activeTexture(weblasTensor.texture);

    // texture dimensions, with padding if necessary
    // see https://github.com/waylonflinn/weblas/blob/master/lib/tensor.js
    const [ h, w ] = this.weblasTensor.shape;
    const pad = this.webgl.getPad(w);

    // target, level, internalformat, x, y, width, height, border
    // https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/copyTexImage2D
    gl.copyTexImage2D(
      this.weblasTensor.texture,
      0,
      gl.RGBA,
      0,
      0,
      (w + pad) / WebGL.COMPONENTS_PER_TEXEL,
      h,
      0
    );
  }
}

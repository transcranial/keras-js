import ndarray from 'ndarray'
import { webgl2, MAX_TEXTURE_SIZE } from './WebGL2'

const checkShape = (data, shape) => {
  if (data.length && shape.length && data.length !== shape.reduce((a, b) => a * b, 1)) {
    throw new Error('Specified shape incompatible with data.')
  }
}

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
    this._type = options.type || Float32Array

    if (data && data.length && (data instanceof this._type || data instanceof Array)) {
      checkShape(data, shape)
      this.tensor = ndarray(data, shape)
      this.tensor = ndarray(new this._type(data), shape)
    } else if (!data.length && shape.length) {
      // if shape present but data not provided, initialize with 0s
      this.tensor = ndarray(new this._type(shape.reduce((a, b) => a * b, 1)), shape)
    } else {
      this.tensor = ndarray(new this._type([]), [])
    }
  }

  /**
   * Replaces data in the underlying ndarray.
   */
  replaceTensorData(data) {
    if (data && data.length && data instanceof this._type) {
      this.tensor.data = data
    } else if (data && data.length && data instanceof Array) {
      this.tensor.data = new this._type(data)
    } else {
      throw new Error('[Tensor] invalid input for replaceTensorData method.')
    }
  }

  createGLTexture() {
    let shape = []
    if (this.tensor.shape.length === 1) {
      shape = [1, this.tensor.shape[0]]
    } else if (this.tensor.shape.length === 2) {
      shape = this.tensor.shape
    } else {
      throw new Error('[Tensor] can only create gpu tensor for 1-D or 2-D shapes only')
    }

    const gl = webgl2.context

    const data = this.tensor.data
    this.glTexture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, this.glTexture)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, shape[1], shape[0], 0, gl.RED, gl.FLOAT, data)

    this.glTextureShape = shape

    // clamp to edge
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    // no interpolation
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  }

  deleteGLTexture() {
    if (this.glTexture) {
      const gl = webgl2.context
      gl.deleteTexture(this.glTexture)
      delete this.glTexture
    }
  }
}

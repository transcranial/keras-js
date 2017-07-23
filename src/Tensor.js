import ndarray from 'ndarray'
import ops from 'ndarray-ops'
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

  /**
   * Creates WebGL2 texture
   * Without args, defaults to gl.TEXTURE_2D and gl.R32F
   */
  createGLTexture(type = '2d', format = 'float') {
    let shape = []
    if (this.tensor.shape.length === 1) {
      shape = [1, this.tensor.shape[0]]
    } else if (this.tensor.shape.length === 2) {
      shape = this.tensor.shape
    } else if (this.tensor.shape.length === 3 && ['2darray', '3d'].includes(type)) {
      shape = this.tensor.shape
    } else {
      throw new Error('[Tensor] cannot create WebGL2 texture.')
    }

    const gl = webgl2.context

    const targetMap = {
      '2d': gl.TEXTURE_2D,
      '2darray': gl.TEXTURE_2D_ARRAY,
      '3d': gl.TEXTURE_3D
    }

    const internalFormatMap = {
      float: gl.R32F,
      int: gl.R32I
    }

    const formatMap = {
      float: gl.RED,
      int: gl.RED_INTEGER
    }

    const typeMap = {
      float: gl.FLOAT,
      int: gl.INT
    }

    this.glTexture = gl.createTexture()
    gl.bindTexture(targetMap[type], this.glTexture)
    if (type === '2d') {
      const data = this.tensor.data

      gl.texImage2D(
        targetMap[type],
        0,
        internalFormatMap[format],
        shape[1],
        shape[0],
        0,
        formatMap[format],
        typeMap[format],
        data
      )
    } else if (type === '2darray' || type === '3d') {
      // must shuffle data layout for webgl
      // data for TEXTURE_2D_ARRAY or TEXTURE_3D laid out sequentially per-slice
      const data = new this._type(this.tensor.data.length)
      const slice = ndarray(new this._type(shape[0] * shape[1]), [shape[0], shape[1]])
      let offset = 0
      for (let i = 0; i < shape[2]; i++) {
        ops.assign(slice, this.tensor.pick(null, null, i))
        data.set(slice.data, offset)
        offset += shape[0] * shape[1]
      }

      gl.texImage3D(
        targetMap[type],
        0,
        internalFormatMap[format],
        shape[1],
        shape[0],
        shape[2],
        0,
        formatMap[format],
        typeMap[format],
        data
      )
    }

    this.glTextureShape = shape

    // clamp to edge
    gl.texParameteri(targetMap[type], gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(targetMap[type], gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    // no interpolation
    gl.texParameteri(targetMap[type], gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(targetMap[type], gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  }

  deleteGLTexture() {
    if (this.glTexture) {
      const gl = webgl2.context
      gl.deleteTexture(this.glTexture)
      delete this.glTexture
    }
  }
}

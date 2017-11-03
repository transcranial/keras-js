import { webgl2, MAX_TEXTURE_SIZE } from './WebGL2'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import squeeze from 'ndarray-squeeze'

/**
 * Function to throw error if specified shape is incompatible with data
 *
 * @param {number[]} data
 * @param {number[]} shape
 */

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
   *
   * @param {(TypedArray|Array)} data
   * @param {number[]} shape
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
   * Replaces data in the underlying ndarray
   *
   * @param {number[]} data
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
   *
   * Without args, defaults to gl.TEXTURE_2D and gl.R32F
   *
   * @param {string} type
   * @param {string} format
   */
  createGLTexture(type = '2d', format = 'float') {
    let shape = []
    if (this.tensor.shape.length === 1) {
      shape = [1, this.tensor.shape[0]]
    } else if (this.tensor.shape.length === 2) {
      shape = this.tensor.shape
    } else if (this.tensor.shape.length === 3 && ['2d_array', '3d'].includes(type)) {
      shape = this.tensor.shape
    } else {
      throw new Error('[Tensor] cannot create WebGL2 texture.')
    }

    const gl = webgl2.context

    const targetMap = {
      '2d': gl.TEXTURE_2D,
      '2d_array': gl.TEXTURE_2D_ARRAY,
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
    } else if (type === '2d_array' || type === '3d') {
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

  /**
   * Deletes WebGLTexture
   */
  deleteGLTexture() {
    if (this.glTexture) {
      const gl = webgl2.context
      gl.deleteTexture(this.glTexture)
      delete this.glTexture
    }
  }

  /**
   * Transfer data from webgl texture on GPU to ndarray on CPU
   */
  transferFromGLTexture() {
    this.tensor = ndarray(new this._type([]), this.glTextureShape)
    this.tensor.data = webgl2.readData(this.glTextureShape)
    if (!this.glTextureIsTiled && this.glTextureShape[0] === 1) {
      // collapse to 1D
      this.tensor = squeeze(this.tensor, [0])
    }
  }

  /**
   * Reshapes data into tiled form
   *
   * @param {number} axis
   */
  reshapeTensorToTiled(axis = -1) {
    if (axis < 0) {
      axis = this.tensor.shape.length + axis
    }

    const normAxisLength = this.tensor.shape[axis]
    const otherAxes = [...this.tensor.shape.slice(0, axis), ...this.tensor.shape.slice(axis + 1)]
    const otherAxesSize = otherAxes.reduce((a, b) => a * b, 1)
    const tiled = ndarray(new this._type(otherAxesSize * normAxisLength), [otherAxesSize, normAxisLength])
    const otherAxesData = ndarray(new this._type(otherAxesSize), otherAxes)
    const otherAxesDataRaveled = ndarray(new this._type(otherAxesSize), [otherAxesSize])
    const axisSlices = Array(this.tensor.shape.length).fill(null)
    for (let n = 0; n < normAxisLength; n++) {
      axisSlices[axis] = n
      ops.assign(otherAxesData, this.tensor.pick(...axisSlices))
      otherAxesDataRaveled.data = otherAxesData.data
      ops.assign(tiled.pick(null, n), otherAxesDataRaveled)
    }

    this.untiledShape = this.tensor.shape
    this.tensor = tiled
    this.glTextureIsTiled = true
  }

  /**
   * Reshapes tiled data into untiled form
   *
   * Called at the end when data is read back from GPU (which is in tiled 2D format from texture)
   *
   * @param {number} axis
   */
  reshapeTensorFromTiled(axis = -1) {
    if (!this.glTextureIsTiled) {
      throw new Error('Tensor is not in tiled format.')
    }
    if (!this.untiledShape) {
      throw new Error('Tensor does not contain untiledShape.')
    }

    if (axis < 0) {
      axis = this.untiledShape.length + axis
    }

    // second axis is the channel, or common, axis
    const channelDataSize = this.tensor.shape[0]
    const channels = this.tensor.shape[1]

    const reshaped = ndarray(new this._type(this.untiledShape.reduce((a, b) => a * b, 1)), this.untiledShape)
    const channelDataRaveled = ndarray(new this._type(channelDataSize), [channelDataSize])
    const untiledChannelShape = [...this.untiledShape.slice(0, axis), ...this.untiledShape.slice(axis + 1)]
    const untiledChannel = ndarray(new this._type(untiledChannelShape.reduce((a, b) => a * b, 1)), untiledChannelShape)
    const axisSlices = Array(this.untiledShape.length).fill(null)
    for (let n = 0; n < channels; n++) {
      ops.assign(channelDataRaveled, this.tensor.pick(null, n))
      untiledChannel.data = channelDataRaveled.data
      axisSlices[axis] = n
      ops.assign(reshaped.pick(...axisSlices), untiledChannel)
    }

    this.tensor = reshaped
  }
}
